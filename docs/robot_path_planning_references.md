# Robot path-planning references and proposed improvements

## 文档目的 / Purpose

本文把当前项目的 CoppeliaSim 路径流程与公开文献、CoppeliaSim/OMPL 官方文档对照，形成可审查的改进依据。它不改变临床规划目标：针尖到达已配准的肺结节，针体与肋骨 AABB、机器人自身、安装板和患者碰撞体保持安全间隙。

This note maps the current CoppeliaSim pipeline to peer-reviewed references and official CoppeliaSim/OMPL documentation. It preserves the clinical objective: reach the registered lung nodule while maintaining clearance from rib AABBs, the robot, the mounting plate, and patient collision geometry.

## 当前实现 / Current implementation

- `trajectory_planner.py` samples straight entry rays in the patient frame and tests a line segment against inflated AABBs with a slab test. The current score is dominated by needle length and, when supplied, angular deviation.
- `create_coppeliasim_scene.py` enumerates needle roll/orientation and IK seeds, then rejects candidates using CoppeliaSim collision pairs and a fixed number of Cartesian samples along `approach -> entry -> target`.
- `coppeliasim_remote.py` moves an IK target through Cartesian waypoints and checks robot/patient, needle/self and mounting collisions at sampled states.
- This is a useful safety preflight, but it is not yet a global robot motion planner. The Cartesian line planner chooses the needle geometry; simIK solves poses locally; the robot's full transit in joint configuration space is not searched by OMPL.

## 建议采用的规划结构 / Recommended planning structure

### 1. Separate free-space transit from constrained insertion

Use a two-stage planner:

1. **Transit** (`initial TCP -> approach`): plan in the UR5 joint configuration space with `simOMPL`, using the measured joint state as start and one or more valid simIK configurations for the approach pose as goals. Use bidirectional RRT-Connect for a fast first feasible solution.
2. **Insertion** (`approach -> entry -> target`): retain the clinical straight needle axis, but generate the robot trajectory with `simIK.generatePath` or repeated IK under orientation/axis constraints. Validate every interpolated state, including the needle sweep, patient AABBs, self-collision and mount collision.
3. **Post-processing**: simplify/shortcut the OMPL transit path, then time-parameterize the resulting joint path with velocity, acceleration and jerk limits (`sim.moveToConfig` or equivalent). Never execute a raw OMPL polyline by unconstrained linear joint interpolation.

This keeps the needle's clinical path interpretable while allowing the arm to route around the scene before insertion.

### 2. Increase feasibility and robustness

- Generate multiple simIK goal configurations using different seeds and reject solutions near joint limits or singularities. A single local IK result can make a geometrically valid needle path appear infeasible.
- Use a state-validity resolution based on maximum joint/Cartesian displacement, rather than always using 100 samples. Densify near obstacles and along high-curvature or high-speed sections.
- Inflate rib AABBs by needle radius **plus** registration, segmentation and calibration uncertainty. If available, use distance queries and a minimum-clearance threshold instead of a binary collision flag only.
- Sample entry points from the segmented skin/chest-wall surface. A radial point at a fixed distance from the target is not a verified anatomical entry point.
- Use a lexicographic or weighted score: collision-free first; then minimum clearance, joint travel, needle length, needle-axis deviation, manipulability/singularity margin and jerk. Selecting only the shortest needle line can produce a difficult or unsafe robot posture.
- For narrow passages or repeated cases, retry with deterministic seeds and a bounded planner portfolio (for example RRT-Connect, KPIECE or LazyPRM where the installed OMPL plugin supports them). Use PRM/LazyPRM when many targets share one patient/robot scene; use RRT*/Informed-RRT*-Connect when path quality, rather than first-solution time, is the priority.

### 3. Collision and execution checks

CoppeliaSim's ordinary collision query is a state query. It does not by itself prove that a fast moving body did not intersect between two sampled states. For the current remote execution path:

- retain discrete checks at every interpolated state;
- add adaptive subdivision until the needle tip, wrist and every link move less than a chosen bound relative to the smallest clearance;
- model the needle as a cylinder/capsule or use a swept segment, not only a point tip;
- where high assurance is required, add continuous/swept collision or proximity checking (for example FCL externally) and compare the minimum clearance with the uncertainty-inflated margin.

### 4. Version-checked CoppeliaSim API path

The locally installed CoppeliaSim manual exposes the following useful calls:

- `simIK.findConfigs(..., {maxTime=..., findMultiple=true, cb=...})` returns several IK configurations and can reject invalid/colliding configurations through `cb`.
- `simIK.generatePath(..., pathPointCount, validationCallback, auxData)` generates the configuration-space path for a straight Cartesian tip motion and validates each proposed configuration through the callback.
- `simOMPL.setStateSpaceForJoints`, `setCollisionPairs`, `setStartState`, `setGoalState`/`setGoalStates`, `setStateValidityCheckingResolution` and `compute` are sufficient for the first joint-space transit implementation. In this installed version, `compute` is equivalent to `solve -> simplifyPath -> interpolatePath -> getPath`.

These calls are version-sensitive; verify the local manual before porting to another CoppeliaSim release.

## 参考文献 / References

### Sampling-based planning and OMPL

1. **Kuffner, J. J.; LaValle, S. M.** “RRT-Connect: An Efficient Approach to Single-Query Path Planning.” *IEEE International Conference on Robotics and Automation*, 2000, pp. 995–1001. DOI: [10.1109/ROBOT.2000.844730](https://doi.org/10.1109/ROBOT.2000.844730). Bidirectional trees make a fast first feasible path for a single start/goal query; suitable for `initial TCP -> approach`.
2. **Karaman, S.; Frazzoli, E.** “Sampling-based Algorithms for Optimal Motion Planning.” *International Journal of Robotics Research*, 30(7), 846–894, 2011. DOI: [10.1177/0278364911406761](https://doi.org/10.1177/0278364911406761). Establishes probabilistic completeness of classical PRM/RRT and asymptotic optimality of PRM*/RRT*; supports an optional optimization stage after RRT-Connect.
3. **Mashayekhi, R.; Idris, M. Y. I.; Anisi, M. H.** “Informed RRT*-Connect: An Asymptotically Optimal Single-Query Path Planning Method.” *IEEE Access*, 8, 19842–19852, 2020. DOI: [10.1109/ACCESS.2020.2969316](https://doi.org/10.1109/ACCESS.2020.2969316). Informed sampling focuses later samples on paths that can improve the first solution.
4. **Elbanhawi, M.; Simic, M.** “Sampling-Based Robot Motion Planning: A Comparative Review.” *IEEE Access*, 2, 56–77, 2014. DOI: [10.1109/ACCESS.2014.2302442](https://doi.org/10.1109/ACCESS.2014.2302442). Review of PRM/RRT variants, narrow-passage behavior, completeness and path-quality trade-offs.
5. **Şucan, I. A.; Moll, M.; Kavraki, L. E.** “The Open Motion Planning Library.” *IEEE Robotics & Automation Magazine*, 19(4), 72–82, 2012. DOI: [10.1109/MRA.2012.2205651](https://doi.org/10.1109/MRA.2012.2205651). OMPL concepts, planners and state-validity/collision interfaces. Official site: [ompl.kavrakilab.org](https://ompl.kavrakilab.org/).

### Trajectory optimization and smoothing

6. **Khatib, O.** “Real-Time Obstacle Avoidance for Manipulators and Mobile Robots.” *International Journal of Robotics Research*, 5(1), 90–98, 1986. DOI: [10.1177/027836498600500106](https://doi.org/10.1177/027836498600500106). Introduces artificial-potential obstacle costs; useful as a local clearance cost, but local minima mean it should refine a feasible seed rather than replace global sampling.
7. **Schulman, J.; Ho, J.; Lee, A.; Awwal, I.; Bradlow, H.; Abbeel, P.** “Motion Planning with Sequential Convex Optimization and Convex Collision Checking.” *International Journal of Robotics Research*, 33(9), 1251–1270, 2014. DOI: [10.1177/0278364914527466](https://doi.org/10.1177/0278364914527466). TrajOpt-style sequential convex optimization can shorten and smooth a collision-free seed while enforcing convexified collision constraints.
8. **Ratliff, N.; Zucker, M.; Bagnell, J. A.; Srinivasa, S.** “CHOMP: Gradient Optimization Techniques for Efficient Motion Planning.” *IEEE International Conference on Robotics and Automation*, 2009. DOI: [10.1109/ROBOT.2009.5152817](https://doi.org/10.1109/ROBOT.2009.5152817). Combines smoothness and obstacle costs; useful for local refinement, with the usual local-minimum caveat.
9. **Zucker, M.; Ratliff, N.; Dragan, A.; et al.** “CHOMP: Covariant Hamiltonian Optimization for Motion Planning.” *International Journal of Robotics Research*, 32(9), 1164–1193, 2013. DOI: [10.1177/0278364913488805](https://doi.org/10.1177/0278364913488805).
10. **Kalakrishnan, M.; Chitta, S.; Theodorou, E.; Pastor, P.; Schaal, S.** “STOMP: Stochastic Trajectory Optimization for Motion Planning.” *IEEE International Conference on Robotics and Automation*, 2011. DOI: [10.1109/ICRA.2011.5980280](https://doi.org/10.1109/ICRA.2011.5980280). Derivative-free stochastic smoothing is useful when collision costs are noisy or non-differentiable.
11. **Geraerts, R.; Overmars, M. H.** “Creating High-quality Paths for Motion Planning.” *International Journal of Robotics Research*, 26(8), 845–863, 2007. DOI: [10.1177/0278364907079280](https://doi.org/10.1177/0278364907079280). Shortcutting and clearance-aware post-processing reduce unnecessary bends while preserving feasibility.
12. **Pham, H.; Pham, Q.-C.** “A New Approach to Time-Optimal Path Parameterization Based on Reachability Analysis.” *IEEE Transactions on Robotics*, 34(3), 645–659, 2018. DOI: [10.1109/TRO.2018.2812913](https://doi.org/10.1109/TRO.2018.2812913). Supports velocity/acceleration constrained time parameterization after geometric planning.
13. **Macfarlane, S.; Croft, E. A.** “Jerk-bounded Robot-trajectory Planning: A Review.” *IEEE Transactions on Robotics and Automation*, 19(1), 42–52, 2003. DOI: [10.1109/TRA.2002.807548](https://doi.org/10.1109/TRA.2002.807548). Motivates explicit jerk limits for executable, smooth robot motion.

### Collision detection and proximity

14. **Redon, S.; Kheddar, A.; Coquillart, S.** “Fast Continuous Collision Detection between Rigid Bodies.” *Computer Graphics Forum*, 21(3), 279–288, 2002. DOI: [10.1111/1467-8659.00587](https://doi.org/10.1111/1467-8659.00587). Continuous swept-motion tests catch collisions that occur between discrete samples.
15. **Pan, J.; Chitta, S.; Manocha, D.** “FCL: A General Purpose Library for Collision and Proximity Queries.” *IEEE International Conference on Robotics and Automation*, 2012, pp. 3859–3866. DOI: [10.1109/ICRA.2012.6225337](https://doi.org/10.1109/ICRA.2012.6225337). Supports discrete/continuous collision, separation distance and penetration depth; relevant if higher-assurance proximity checks are added outside CoppeliaSim.

### Medical needle planning

16. **Alterovitz, R.; Branicky, M.; Goldberg, K.** “Motion Planning Under Uncertainty for Image-guided Medical Needle Steering.” *International Journal of Robotics Research*, 27(11–12), 1361–1374, 2008. DOI: [10.1177/0278364908097661](https://doi.org/10.1177/0278364908097661). Models steering and obstacle uncertainty probabilistically; supports treating registration/segmentation error as a planning margin or success-probability term.
17. **Reed, K. B.; Kallem, V.; Alterovitz, R.; Goldberg, K.; Okamura, A. M.; Cowan, N. J.** “Integrated Planning and Image-Guided Control for Planar Needle Steering.” *IEEE/RAS-EMBS International Conference on Biomedical Robotics and Biomechatronics*, 2008, pp. 819–824. DOI: [10.1109/BIOROB.2008.4762833](https://doi.org/10.1109/BIOROB.2008.4762833). Combines a stochastic motion roadmap with image feedback, illustrating the value of planning plus closed-loop verification.
18. **Vrooijink, G. J.; Abayazid, M.; Patil, S.; Alterovitz, R.; Misra, S.** “Needle Path Planning and Steering in a Three-dimensional Non-static Environment Using Two-dimensional Ultrasound Images.” *International Journal of Robotics Research*, 33(10), 1361–1374, 2014. DOI: [10.1177/0278364914526627](https://doi.org/10.1177/0278364914526627). Closed-loop RRT replanning addresses moving targets/obstacles; relevant if respiratory motion is later modeled.
19. **DiMaio, S. P.; Salcudean, S. E.** “Needle Insertion Modeling and Simulation.” *IEEE Transactions on Robotics and Automation*, 19(5), 864–875, 2003. DOI: [10.1109/TRA.2003.817044](https://doi.org/10.1109/TRA.2003.817044). Provides a basis for later needle deflection and insertion-force modeling; the present project currently assumes a rigid straight needle.
20. **Halder, K.; Orlando, M. F.** “Path Planning for Robot Assisted Steerable Bevel-Tip Needle in 3D Dynamic Environment.” *IEEE Robotics and Automation Letters*, 10(6), 5887–5894, 2025. DOI: [10.1109/LRA.2025.3562010](https://doi.org/10.1109/LRA.2025.3562010). A recent needle-specific RRT-based planning reference for dynamic 3-D scenes; it is relevant to a future respiratory-motion extension, but the present project still uses a rigid straight needle and static CT geometry.
21. **Duindam, V.; Xu, J.; Alterovitz, R.; Sastry, S.; Goldberg, K.** “Three-dimensional Motion Planning Algorithms for Steerable Needles Using Inverse Kinematics.” *International Journal of Robotics Research*, 29(7), 789–800, 2010. DOI: [10.1177/0278364909352202](https://doi.org/10.1177/0278364909352202). A 3-D inverse-kinematics planning reference for steerable needles; useful if the project later replaces the rigid straight needle assumption with a flexible/nonholonomic model.
22. **Fu, M.; Kuntz, A.; Webster, R. J. III; Alterovitz, R.** “Safe Motion Planning for Steerable Needles Using Cost Maps Automatically Extracted from Pulmonary Images.” *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2018, pp. 4942–4949. DOI: [10.1109/IROS.2018.8593407](https://doi.org/10.1109/IROS.2018.8593407). Extracts pulmonary structures and risk costs from CT into a planning cost map; directly supports replacing a binary rib-AABB test with a clearance/risk field when pleura, vessels or bronchi are available.

### CoppeliaSim official documentation

23. **CoppeliaSim simOMPL plugin/manual:** [manual.coppeliarobotics.com/en/simOMPL.htm](https://manual.coppeliarobotics.com/en/simOMPL.htm). Use the installed version's exact API names for state spaces, collision pairs, solve, path retrieval and simplification.
24. **CoppeliaSim simIK plugin/manual:** [manual.coppeliarobotics.com/en/simIK.htm](https://manual.coppeliarobotics.com/en/simIK.htm). Use for multiple IK seeds, constrained tip poses and IK path generation; exact option names are version-dependent.
25. **CoppeliaSim motion-profile API (`sim.moveToConfig`):** [manual.coppeliarobotics.com/en/simMoveToConfig.htm](https://manual.coppeliarobotics.com/en/simMoveToConfig.htm). Use for velocity/acceleration/jerk-limited execution of a validated joint path.
26. **CoppeliaSim API and collision functions:** [manual.coppeliarobotics.com/en/apiFunctions.htm](https://manual.coppeliarobotics.com/en/apiFunctions.htm). `sim.checkCollision` remains a discrete state query; pair definitions must include patient, robot/self, needle and mount geometry.

## 采用顺序 / Implementation order

1. **First:** multi-goal IK + OMPL RRT-Connect for `initial TCP -> approach`, with full collision pairs and start/goal validity checks.
2. **Second:** adaptive state interpolation and clearance checks for the insertion polyline; uncertainty-inflated AABBs and surface-based entry sampling.
3. **Third:** OMPL shortcut/simplification, then jerk-limited time parameterization.
4. **Fourth:** optional TrajOpt/CHOMP/STOMP local refinement and RRT*/Informed-RRT*-Connect comparison when path quality or repeated failures justify the extra computation.
5. **Evaluation:** report first-solution success rate, end-to-target error, minimum clearance, IK failure rate, collision rejection rate, joint travel, execution time and jerk. Compare the current fixed-100-waypoint baseline against each change on the same registered cases and random seeds.

## 限制 / Limitations

- CoppeliaSim's available OMPL algorithms and simIK options vary by release; verify the local plugin documentation before coding against an enum or function name.
- AABB obstacles are conservative and can either reject safe paths or miss geometry outside the segmented boxes. They are not a substitute for validated rib surfaces or volumetric masks.
- CHOMP, TrajOpt and potential-field methods are local refinements and may get trapped in local minima; they should receive a collision-free global seed.
- The cited needle-steering papers often model flexible bevel-tip needles and dynamic imaging. The current project uses a rigid straight needle and static CT/AABB geometry, so those methods motivate uncertainty/feedback extensions rather than drop-in replacements.
- Continuous collision detection is not automatically provided by a fixed list of `sim.checkCollision` calls. A high-assurance implementation must either use a supported swept-volume/proximity method or prove a sufficiently small interpolation bound.
