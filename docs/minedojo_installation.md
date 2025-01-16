# MineDojo Installation Steps and Troubleshooting

## MineDojo Installation Steps

1. Install the appropriate version of Java as per the [official documentation](https://docs.minedojo.org/sections/getting_started/install.html#prerequisites).
    - After installation, test the installation with the command `java -version`. If successful, it will output something similar to:
        ```bash
        openjdk version "1.8.0_322"
        OpenJDK Runtime Environment (Temurin)(build 1.8.0_322-b06)
        OpenJDK 64-Bit Server VM (Temurin)(build 25.322-b06, mixed mode)
        ```
    - When running the command `sudo apt install xvfb xserver-xephyr vnc4server python-opengl ffmpeg`, if errors occur for certain packages, you can ignore them as long as `xvfb` installs successfully.

2. Activate the conda environment:
    ```bash
    conda activate ls_imagine
    ```

3. Install `minedojo`:
    ```bash
    pip install minedojo
    ```
   - If an error occurs while installing `gym`, such as the following:
        - <details>
            <summary>Click to expand</summary>

            ```bash
            Collecting gym==0.21.0
            Using cached gym-0.21.0.tar.gz (1.5 MB)
            Preparing metadata (setup.py) ... error
            error: subprocess-exited-with-error

            × python setup.py egg_info did not run successfully.
            │ exit code: 1
            ╰─> [1 lines of output]
                error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.
                [end of output]
            
            note: This error originates from a subprocess, and is likely not a problem with pip.
            error: metadata-generation-failed
            
            × Encountered error while generating package metadata.
            ╰─> See above for output.
            
            note: This is an issue with the package mentioned above, not pip.
            hint: See above for details.
            ```
        </details>

        - **Solution**:
            ```bash
            pip install setuptools==65.5.0
            pip install --user wheel==0.38.0
            ```

    - Another possible error:
        - <details>
            <summary>Click to expand</summary>

            ```bash
            ERROR: Could not find a version that satisfies the requirement gym==0.21.0 (from versions: 0.0.2, 0.0.3, 0.0.4, 0.0.5, 0.0.6, 0.0.7, 0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.1.4, 0.1.5, 0.1.6, 0.1.7, 0.2.0, 0.2.1, 0.2.2, 0.2.3, 0.2.4, 0.2.5, 0.2.6, 0.2.7, 0.2.8, 0.2.9, 0.2.10, 0.2.11, 0.2.12, 0.3.0, 0.4.0, 0.4.1, 0.4.2, 0.4.3, 0.4.4, 0.4.5, 0.4.6, 0.4.8, 0.4.9, 0.4.10, 0.5.0, 0.5.1, 0.5.2, 0.5.3, 0.5.4, 0.5.5, 0.5.6, 0.5.7, 0.6.0, 0.7.0, 0.7.1, 0.7.2, 0.7.3, 0.7.4, 0.8.0.dev0, 0.8.0, 0.8.1, 0.8.2, 0.9.0, 0.9.1, 0.9.2, 0.9.3, 0.9.4, 0.9.5, 0.9.6, 0.9.7, 0.10.0, 0.10.1, 0.10.2, 0.10.3, 0.10.4, 0.10.5, 0.10.8, 0.10.9, 0.10.11, 0.11.0, 0.12.0, 0.12.1, 0.12.4, 0.12.5, 0.12.6, 0.13.0, 0.13.1, 0.14.0, 0.15.3, 0.15.4, 0.15.6, 0.15.7, 0.16.0, 0.17.0, 0.17.1, 0.17.2, 0.17.3, 0.18.0, 0.18.3, 0.19.0, 0.20.0, 0.21.0, 0.22.0, 0.23.0, 0.23.1, 0.24.0, 0.24.1, 0.25.0, 0.25.1, 0.25.2, 0.26.0, 0.26.1, 0.26.2)
            ERROR: No matching distribution found for gym==0.21.0
            ```
        </details>

        - **Solution**:
            ```bash
            pip install pip==24.0
            ```

4. Install `MineCLIP`:
   - It is recommended to clone the repository locally before installation:
        ```bash
        git clone https://github.com/MineDojo/MineCLIP.git
        cd MineCLIP
        pip install -e .
        ```

5. Use test code to check whether the installation was successful:
    ```python
    # test_minedojo.py
    import minedojo

    env = minedojo.make(
        task_id="harvest_wool_with_shears_and_sheep",
        image_size=(160, 256)
    )

    obs = env.reset()
    for i in range(50):
        print(f"===== Step: {i+1} =====")
        act = env.action_space.no_op()
        act[0] = 1    # forward/backward
        if i % 10 == 0:
            act[2] = 1    # jump
        obs, reward, done, info = env.step(act)
        
    env.close()
    ```
    - Run the following command. If the step count outputs without errors, the installation is successful:
        ```bash
        MINEDOJO_HEADLESS=1 python test_minedojo.py
        ```
    - If an error occurs, refer to [the next section](#troubleshooting-common-errors-after-installation).

## Troubleshooting Common Errors After Installation

> - Errors usually occur at the `env.reset()` step.

1. Gradle download or extraction error:
    - <details>
        <summary>Click to expand</summary>

        ```bash
        HELLO
        Downloading https://services.gradle.org/distributions/gradle-4.10.2-all.zip
        ........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
        Unzipping ~/.gradle/wrapper/dists/gradle-4.10.2-all/9fahxiiecdb76a5g3aw9oi8rv/gradle-4.10.2-all.zip to ~/.gradle/wrapper/dists/gradle-4.10.2-all/9fahxiiecdb76a5g3aw9oi8rv
        Exception in thread "main" java.util.zip.ZipException: zip END header not found
        ```
        </details>

   - The issue usually arises from an unstable download or incomplete file causing an extraction error.
   - **Solution**: Manually download the [Gradle package](https://services.gradle.org/distributions/gradle-4.10.2-all.zip) and place the downloaded `.zip` file in the directory `~/.gradle/wrapper/dists/gradle-4.10.2-all/9fahxiiecdb76a5g3aw9oi8rv/`.
   - Alternatively, you can refer to [this guide](https://luo3.org.cn/posts/snippets/replace-gradle-repositories-with-aliyun-mirrors/) to configure a mirror for faster and more stable downloads.

2. Error: Could not find `MixinGradle:dcfaf61`
    - <details>
        <summary>Click to expand</summary>

        ```bash
        HELLO
        Starting a Gradle Daemon (subsequent builds will be faster)
        FAILURE: Build failed with an exception.
        * What went wrong:
        A problem occurred configuring root project 'Minecraft'.
        > Could not resolve all artifacts for configuration ':classpath'.
            > Could not find com.github.SpongePowered:MixinGradle:dcfaf61.
        ```
        </details>

    - Refer to the [GitHub issue](https://github.com/MineDojo/MineDojo/issues/113) for details.
    - Clone the repository `https://github.com/verityw/MixinGradle-dcfaf61` to a local directory (e.g., clone it to `~/workspace/`):
        ```bash
        git clone https://github.com/verityw/MixinGradle-dcfaf61 ~/workspace/
        ```

    - Locate the `build.gradle` file to modify. This file is usually found in the `minedojo` installation path, for example:
      `~/.conda/envs/minedojo/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build.gradle`.

    - Replace the following section:
        ```gradle
        buildscript {
            repositories {

                maven { url 'https://jitpack.io' }
                jcenter()
                mavenCentral()
                maven {
                    name = "forge"
                    url = "https://maven.minecraftforge.net/"
                }
                maven {
                    name = "sonatype"
                    url = "https://oss.sonatype.org/content/repositories/snapshots/"
                }
            }
            dependencies {
                classpath 'org.ow2.asm:asm:6.0'
                classpath('com.github.SpongePowered:MixinGradle:dcfaf61'){ // 0.6
                    // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
                    // and putting mixin here places it before forge in the class loader
                    exclude group: 'org.ow2.asm', module: 'asm-debug-all'
                }

                classpath 'com.github.yunfanjiang:ForgeGradle:FG_2.2_patched-SNAPSHOT'
            }
        }
        ```

        With the following:
        ```gradle
        buildscript {
            repositories {

                maven { url 'https://jitpack.io' }
                jcenter()
                mavenCentral()
                maven {
                    url "file:~/workspace" // Local directory where the repository was cloned
                }
                maven {
                    name = "forge"
                    url = "https://maven.minecraftforge.net/"
                }
                maven {
                    name = "sonatype"
                    url = "https://oss.sonatype.org/content/repositories/snapshots/"
                }
            }
            dependencies {
                classpath 'org.ow2.asm:asm:6.0'
                // classpath('com.github.SpongePowered:MixinGradle:dcfaf61'){ // 0.6
                //     // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
                //     // and putting mixin here places it before forge in the class loader
                //     exclude group: 'org.ow2.asm', module: 'asm-debug-all'
                // }
                classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
                    // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
                    // and putting mixin here places it before forge in the class loader
                    exclude group: 'org.ow2.asm', module: 'asm-debug-all'
                }

                classpath 'com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT'
            }
        }
        ```

3. If you encounter an error similar to the following:

   - <details>
        <summary>Click to expand</summary>

        ```bash
        Traceback (most recent call last):
          ...
          File "/opt/conda/lib/python3.9/site-packages/minedojo/tasks/meta/base.py", line 87, in reset
            obs = self.env.reset()
          File "/opt/conda/lib/python3.9/site-packages/minedojo/sim/sim.py", line 419, in reset
            raw_obs = self._bridge_env.reset(episode_id, [xml])[0]
          File "/opt/conda/lib/python3.9/site-packages/minedojo/sim/bridge/bridge_env/bridge_env.py", line 72, in reset
            self._setup_instances()
          File "/opt/conda/lib/python3.9/site-packages/minedojo/sim/bridge/bridge_env/bridge_env.py", line 157, in _setup_instances
            self._instances.extend([f.result() for f in instance_futures])
          File "/opt/conda/lib/python3.9/site-packages/minedojo/sim/bridge/bridge_env/bridge_env.py", line 157, in <listcomp>
            self._instances.extend([f.result() for f in instance_futures])
          File "/opt/conda/lib/python3.9/concurrent/futures/_base.py", line 438, in result
            return self.__get_result()
          File "/opt/conda/lib/python3.9/concurrent/futures/_base.py", line 390, in __get_result
            raise self._exception
          File "/opt/conda/lib/python3.9/concurrent/futures/thread.py", line 52, in run
            result = self.fn(*self.args, **self.kwargs)
          File "/opt/conda/lib/python3.9/site-packages/minedojo/sim/bridge/bridge_env/bridge_env.py", line 177, in _get_new_instance
            instance.launch(replaceable=self._is_fault_tolerant)
          File "/opt/conda/lib/python3.9/site-packages/minedojo/sim/bridge/mc_instance/instance.py", line 201, in launch
            raise EOFError(
        EOFError: /tmp/tmpwt5vhp9h/Minecraft
        # Configuration file
        # Autogenerated from command-line options
        
        malmoports {
          I:portOverride=11992
        }
        malmoscore {
          I:policy=0
        }
        
        malmoseed {
          I:seed=1836862008
        }
        
        runtype {
          B:replaceable=true
        }
        
        envtype {
          B:env=true
        }
        
        /tmp/tmpwt5vhp9h/Minecraft/run
        
        HELLO
        
        FAILURE: Build failed with an exception.
        
        * What went wrong:
        A problem occurred configuring root project 'Minecraft'.
        > Could not resolve all artifacts for configuration ':classpath'.
           > Could not resolve com.github.johnrengelman.shadow:com.github.johnrengelman.shadow.gradle.plugin:1.2.4.
             Required by:
                 project :
              > Could not resolve com.github.johnrengelman.shadow:com.github.johnrengelman.shadow.gradle.plugin:1.2.4.
                 > Could not get resource 'https://jcenter.bintray.com/com/github/johnrengelman/shadow/com.github.johnrengelman.shadow.gradle.plugin/1.2.4/com.github.johnrengelman.shadow.gradle.plugin-1.2.4.pom'.
                    > Could not HEAD 'https://jcenter.bintray.com/com/github/johnrengelman/shadow/com.github.johnrengelman.shadow.gradle.plugin/1.2.4/com.github.johnrengelman.shadow.gradle.plugin-1.2.4.pom'.
                       > Read timed out
        
        * Try:
        Run with --info or --debug option to get more log output. Run with --scan to get full insights.
        
        * Exception is:
        org.gradle.api.ProjectConfigurationException: A problem occurred configuring root project 'Minecraft'.
                at org.gradle.configuration.project.LifecycleProjectEvaluator.wrapException(LifecycleProjectEvaluator.java:79)
                at org.gradle.configuration.project.LifecycleProjectEvaluator.addConfigurationFailure(LifecycleProjectEvaluator.java:73)
                at org.gradle.configuration.project.LifecycleProjectEvaluator.access$400(LifecycleProjectEvaluator.java:54)
                at org.gradle.configuration.project.LifecycleProjectEvaluator$EvaluateProject.run(LifecycleProjectEvaluator.java:107)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:300)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:292)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.execute(DefaultBuildOperationExecutor.java:174)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.run(DefaultBuildOperationExecutor.java:90)
                at org.gradle.internal.operations.DelegatingBuildOperationExecutor.run(DelegatingBuildOperationExecutor.java:31)
                at org.gradle.configuration.project.LifecycleProjectEvaluator.evaluate(LifecycleProjectEvaluator.java:68)
                at org.gradle.api.internal.project.DefaultProject.evaluate(DefaultProject.java:687)
                at org.gradle.api.internal.project.DefaultProject.evaluate(DefaultProject.java:140)
                at org.gradle.execution.TaskPathProjectEvaluator.configure(TaskPathProjectEvaluator.java:35)
                at org.gradle.execution.TaskPathProjectEvaluator.configureHierarchy(TaskPathProjectEvaluator.java:60)
                at org.gradle.configuration.DefaultBuildConfigurer.configure(DefaultBuildConfigurer.java:41)
                at org.gradle.initialization.DefaultGradleLauncher$ConfigureBuild.run(DefaultGradleLauncher.java:274)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:300)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:292)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.execute(DefaultBuildOperationExecutor.java:174)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.run(DefaultBuildOperationExecutor.java:90)
                at org.gradle.internal.operations.DelegatingBuildOperationExecutor.run(DelegatingBuildOperationExecutor.java:31)
                at org.gradle.initialization.DefaultGradleLauncher.configureBuild(DefaultGradleLauncher.java:182)
                at org.gradle.initialization.DefaultGradleLauncher.doBuildStages(DefaultGradleLauncher.java:141)
                at org.gradle.initialization.DefaultGradleLauncher.executeTasks(DefaultGradleLauncher.java:124)
                at org.gradle.internal.invocation.GradleBuildController$1.call(GradleBuildController.java:77)
                at org.gradle.internal.invocation.GradleBuildController$1.call(GradleBuildController.java:74)
                at org.gradle.internal.work.DefaultWorkerLeaseService.withLocks(DefaultWorkerLeaseService.java:154)
                at org.gradle.internal.work.StopShieldingWorkerLeaseService.withLocks(StopShieldingWorkerLeaseService.java:38)
                at org.gradle.internal.invocation.GradleBuildController.doBuild(GradleBuildController.java:96)
                at org.gradle.internal.invocation.GradleBuildController.run(GradleBuildController.java:74)
                at org.gradle.tooling.internal.provider.ExecuteBuildActionRunner.run(ExecuteBuildActionRunner.java:28)
                at org.gradle.launcher.exec.ChainingBuildActionRunner.run(ChainingBuildActionRunner.java:35)
                at org.gradle.tooling.internal.provider.ValidatingBuildActionRunner.run(ValidatingBuildActionRunner.java:32)
                at org.gradle.launcher.exec.RunAsBuildOperationBuildActionRunner$3.run(RunAsBuildOperationBuildActionRunner.java:50)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:300)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:292)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.execute(DefaultBuildOperationExecutor.java:174)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.run(DefaultBuildOperationExecutor.java:90)
                at org.gradle.internal.operations.DelegatingBuildOperationExecutor.run(DelegatingBuildOperationExecutor.java:31)
                at org.gradle.launcher.exec.RunAsBuildOperationBuildActionRunner.run(RunAsBuildOperationBuildActionRunner.java:45)
                at org.gradle.tooling.internal.provider.SubscribableBuildActionRunner.run(SubscribableBuildActionRunner.java:51)
                at org.gradle.launcher.exec.InProcessBuildActionExecuter$1.transform(InProcessBuildActionExecuter.java:47)
                at org.gradle.launcher.exec.InProcessBuildActionExecuter$1.transform(InProcessBuildActionExecuter.java:44)
                at org.gradle.composite.internal.DefaultRootBuildState.run(DefaultRootBuildState.java:79)
                at org.gradle.launcher.exec.InProcessBuildActionExecuter.execute(InProcessBuildActionExecuter.java:44)
                at org.gradle.launcher.exec.InProcessBuildActionExecuter.execute(InProcessBuildActionExecuter.java:30)
                at org.gradle.launcher.exec.BuildTreeScopeBuildActionExecuter.execute(BuildTreeScopeBuildActionExecuter.java:39)
                at org.gradle.launcher.exec.BuildTreeScopeBuildActionExecuter.execute(BuildTreeScopeBuildActionExecuter.java:25)
                at org.gradle.tooling.internal.provider.ContinuousBuildActionExecuter.execute(ContinuousBuildActionExecuter.java:80)
                at org.gradle.tooling.internal.provider.ContinuousBuildActionExecuter.execute(ContinuousBuildActionExecuter.java:53)
                at org.gradle.tooling.internal.provider.ServicesSetupBuildActionExecuter.execute(ServicesSetupBuildActionExecuter.java:62)
                at org.gradle.tooling.internal.provider.ServicesSetupBuildActionExecuter.execute(ServicesSetupBuildActionExecuter.java:34)
                at org.gradle.tooling.internal.provider.GradleThreadBuildActionExecuter.execute(GradleThreadBuildActionExecuter.java:36)
                at org.gradle.tooling.internal.provider.GradleThreadBuildActionExecuter.execute(GradleThreadBuildActionExecuter.java:25)
                at org.gradle.tooling.internal.provider.ParallelismConfigurationBuildActionExecuter.execute(ParallelismConfigurationBuildActionExecuter.java:43)
                at org.gradle.tooling.internal.provider.ParallelismConfigurationBuildActionExecuter.execute(ParallelismConfigurationBuildActionExecuter.java:29)
                at org.gradle.tooling.internal.provider.StartParamsValidatingActionExecuter.execute(StartParamsValidatingActionExecuter.java:59)
                at org.gradle.tooling.internal.provider.StartParamsValidatingActionExecuter.execute(StartParamsValidatingActionExecuter.java:31)
                at org.gradle.tooling.internal.provider.SessionFailureReportingActionExecuter.execute(SessionFailureReportingActionExecuter.java:59)
                at org.gradle.tooling.internal.provider.SessionFailureReportingActionExecuter.execute(SessionFailureReportingActionExecuter.java:44)
                at org.gradle.tooling.internal.provider.SetupLoggingActionExecuter.execute(SetupLoggingActionExecuter.java:46)
                at org.gradle.tooling.internal.provider.SetupLoggingActionExecuter.execute(SetupLoggingActionExecuter.java:30)
                at org.gradle.launcher.daemon.server.exec.ExecuteBuild.doBuild(ExecuteBuild.java:67)
                at org.gradle.launcher.daemon.server.exec.BuildCommandOnly.execute(BuildCommandOnly.java:36)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.WatchForDisconnection.execute(WatchForDisconnection.java:37)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.ResetDeprecationLogger.execute(ResetDeprecationLogger.java:26)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.RequestStopIfSingleUsedDaemon.execute(RequestStopIfSingleUsedDaemon.java:34)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.ForwardClientInput$2.call(ForwardClientInput.java:74)
                at org.gradle.launcher.daemon.server.exec.ForwardClientInput$2.call(ForwardClientInput.java:72)
                at org.gradle.util.Swapper.swap(Swapper.java:38)
                at org.gradle.launcher.daemon.server.exec.ForwardClientInput.execute(ForwardClientInput.java:72)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.LogAndCheckHealth.execute(LogAndCheckHealth.java:55)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.LogToClient.doBuild(LogToClient.java:62)
                at org.gradle.launcher.daemon.server.exec.BuildCommandOnly.execute(BuildCommandOnly.java:36)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.EstablishBuildEnvironment.doBuild(EstablishBuildEnvironment.java:81)
                at org.gradle.launcher.daemon.server.exec.BuildCommandOnly.execute(BuildCommandOnly.java:36)
                at org.gradle.launcher.daemon.server.api.DaemonCommandExecution.proceed(DaemonCommandExecution.java:122)
                at org.gradle.launcher.daemon.server.exec.StartBuildOrRespondWithBusy$1.run(StartBuildOrRespondWithBusy.java:50)
                at org.gradle.launcher.daemon.server.DaemonStateCoordinator$1.run(DaemonStateCoordinator.java:295)
                at org.gradle.internal.concurrent.ExecutorPolicy$CatchAndRecordFailures.onExecute(ExecutorPolicy.java:63)
                at org.gradle.internal.concurrent.ManagedExecutorImpl$1.run(ManagedExecutorImpl.java:46)
                at org.gradle.internal.concurrent.ThreadFactoryImpl$ManagedThreadRunnable.run(ThreadFactoryImpl.java:55)
        Caused by: org.gradle.api.internal.artifacts.ivyservice.DefaultLenientConfiguration$ArtifactResolveException: Could not resolve all artifacts for configuration ':classpath'.
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration.rethrowFailure(DefaultConfiguration.java:1054)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration.access$1700(DefaultConfiguration.java:123)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration$ConfigurationArtifactCollection.ensureResolved(DefaultConfiguration.java:1489)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration$ConfigurationArtifactCollection.getArtifacts(DefaultConfiguration.java:1461)
                at org.gradle.composite.internal.CompositeBuildClassPathInitializer.execute(CompositeBuildClassPathInitializer.java:45)
                at org.gradle.composite.internal.CompositeBuildClassPathInitializer.execute(CompositeBuildClassPathInitializer.java:32)
                at org.gradle.api.internal.initialization.DefaultScriptClassPathResolver.resolveClassPath(DefaultScriptClassPathResolver.java:37)
                at org.gradle.api.internal.initialization.DefaultScriptHandler.getScriptClassPath(DefaultScriptHandler.java:74)
                at org.gradle.plugin.use.internal.DefaultPluginRequestApplicator.defineScriptHandlerClassScope(DefaultPluginRequestApplicator.java:204)
                at org.gradle.plugin.use.internal.DefaultPluginRequestApplicator.applyPlugins(DefaultPluginRequestApplicator.java:140)
                at org.gradle.configuration.DefaultScriptPluginFactory$ScriptPluginImpl.apply(DefaultScriptPluginFactory.java:186)
                at org.gradle.configuration.BuildOperationScriptPlugin$1$1.run(BuildOperationScriptPlugin.java:69)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:300)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:292)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.execute(DefaultBuildOperationExecutor.java:174)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.run(DefaultBuildOperationExecutor.java:90)
                at org.gradle.internal.operations.DelegatingBuildOperationExecutor.run(DelegatingBuildOperationExecutor.java:31)
                at org.gradle.configuration.BuildOperationScriptPlugin$1.execute(BuildOperationScriptPlugin.java:66)
                at org.gradle.configuration.BuildOperationScriptPlugin$1.execute(BuildOperationScriptPlugin.java:63)
                at org.gradle.configuration.internal.DefaultUserCodeApplicationContext.apply(DefaultUserCodeApplicationContext.java:48)
                at org.gradle.configuration.BuildOperationScriptPlugin.apply(BuildOperationScriptPlugin.java:63)
                at org.gradle.configuration.project.BuildScriptProcessor.execute(BuildScriptProcessor.java:41)
                at org.gradle.configuration.project.BuildScriptProcessor.execute(BuildScriptProcessor.java:26)
                at org.gradle.configuration.project.ConfigureActionsProjectEvaluator.evaluate(ConfigureActionsProjectEvaluator.java:34)
                at org.gradle.configuration.project.LifecycleProjectEvaluator$EvaluateProject.run(LifecycleProjectEvaluator.java:105)
                ... 85 more
        Caused by: org.gradle.internal.resolve.ModuleVersionResolveException: Could not resolve com.github.johnrengelman.shadow:com.github.johnrengelman.shadow.gradle.plugin:1.2.4.
        Required by:
            project :
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.RepositoryChainComponentMetaDataResolver.resolveModule(RepositoryChainComponentMetaDataResolver.java:103)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.RepositoryChainComponentMetaDataResolver.resolve(RepositoryChainComponentMetaDataResolver.java:63)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.ComponentResolversChain$ComponentMetaDataResolverChain.resolve(ComponentResolversChain.java:94)
                at org.gradle.api.internal.artifacts.ivyservice.clientmodule.ClientModuleResolver.resolve(ClientModuleResolver.java:62)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.ComponentState.resolve(ComponentState.java:208)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.ComponentState.resolve(ComponentState.java:196)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.ComponentState.getMetadata(ComponentState.java:152)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.EdgeState.calculateTargetConfigurations(EdgeState.java:156)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.EdgeState.attachToTargetConfigurations(EdgeState.java:112)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.DependencyGraphBuilder.attachToTargetRevisionsSerially(DependencyGraphBuilder.java:315)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.DependencyGraphBuilder.resolveEdges(DependencyGraphBuilder.java:202)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.DependencyGraphBuilder.traverseGraph(DependencyGraphBuilder.java:155)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.graph.builder.DependencyGraphBuilder.resolve(DependencyGraphBuilder.java:126)
                at org.gradle.api.internal.artifacts.ivyservice.resolveengine.DefaultArtifactDependencyResolver.resolve(DefaultArtifactDependencyResolver.java:123)
                at org.gradle.api.internal.artifacts.ivyservice.DefaultConfigurationResolver.resolveGraph(DefaultConfigurationResolver.java:167)
                at org.gradle.api.internal.artifacts.ivyservice.ShortCircuitEmptyConfigurationResolver.resolveGraph(ShortCircuitEmptyConfigurationResolver.java:89)
                at org.gradle.api.internal.artifacts.ivyservice.ErrorHandlingConfigurationResolver.resolveGraph(ErrorHandlingConfigurationResolver.java:73)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration$5.run(DefaultConfiguration.java:533)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:300)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$RunnableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:292)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.execute(DefaultBuildOperationExecutor.java:174)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.run(DefaultBuildOperationExecutor.java:90)
                at org.gradle.internal.operations.DelegatingBuildOperationExecutor.run(DelegatingBuildOperationExecutor.java:31)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration.resolveGraphIfRequired(DefaultConfiguration.java:524)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration.resolveToStateOrLater(DefaultConfiguration.java:509)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration.access$1800(DefaultConfiguration.java:123)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration$ConfigurationFileCollection.getSelectedArtifacts(DefaultConfiguration.java:1037)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration$ConfigurationFileCollection.access$3100(DefaultConfiguration.java:971)
                at org.gradle.api.internal.artifacts.configurations.DefaultConfiguration$ConfigurationArtifactCollection.ensureResolved(DefaultConfiguration.java:1483)
                ... 107 more
        Caused by: org.gradle.internal.resolve.ModuleVersionResolveException: Could not resolve com.github.johnrengelman.shadow:com.github.johnrengelman.shadow.gradle.plugin:1.2.4.
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.ErrorHandlingModuleComponentRepository$ErrorHandlingModuleComponentRepositoryAccess.resolveComponentMetaData(ErrorHandlingModuleComponentRepository.java:141)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.ComponentMetaDataResolveState.process(ComponentMetaDataResolveState.java:75)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.ComponentMetaDataResolveState.resolve(ComponentMetaDataResolveState.java:63)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.RepositoryChainComponentMetaDataResolver.findBestMatch(RepositoryChainComponentMetaDataResolver.java:138)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.RepositoryChainComponentMetaDataResolver.findBestMatch(RepositoryChainComponentMetaDataResolver.java:119)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.RepositoryChainComponentMetaDataResolver.resolveModule(RepositoryChainComponentMetaDataResolver.java:92)
                ... 135 more
        Caused by: org.gradle.api.resources.ResourceException: Could not get resource 'https://jcenter.bintray.com/com/github/johnrengelman/shadow/com.github.johnrengelman.shadow.gradle.plugin/1.2.4/com.github.johnrengelman.shadow.gradle.plugin-1.2.4.pom'.
                at org.gradle.internal.resource.ResourceExceptions.failure(ResourceExceptions.java:74)
                at org.gradle.internal.resource.ResourceExceptions.getFailed(ResourceExceptions.java:57)
                at org.gradle.api.internal.artifacts.repositories.resolver.DefaultExternalResourceArtifactResolver.downloadByCoords(DefaultExternalResourceArtifactResolver.java:138)
                at org.gradle.api.internal.artifacts.repositories.resolver.DefaultExternalResourceArtifactResolver.downloadStaticResource(DefaultExternalResourceArtifactResolver.java:97)
                at org.gradle.api.internal.artifacts.repositories.resolver.DefaultExternalResourceArtifactResolver.resolveArtifact(DefaultExternalResourceArtifactResolver.java:64)
                at org.gradle.api.internal.artifacts.repositories.metadata.AbstractRepositoryMetadataSource.parseMetaDataFromArtifact(AbstractRepositoryMetadataSource.java:69)
                at org.gradle.api.internal.artifacts.repositories.metadata.AbstractRepositoryMetadataSource.create(AbstractRepositoryMetadataSource.java:59)
                at org.gradle.api.internal.artifacts.repositories.resolver.ExternalResourceResolver.resolveStaticDependency(ExternalResourceResolver.java:244)
                at org.gradle.api.internal.artifacts.repositories.resolver.MavenResolver.doResolveComponentMetaData(MavenResolver.java:127)
                at org.gradle.api.internal.artifacts.repositories.resolver.ExternalResourceResolver$RemoteRepositoryAccess.resolveComponentMetaData(ExternalResourceResolver.java:445)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.CachingModuleComponentRepository$ResolveAndCacheRepositoryAccess.resolveComponentMetaData(CachingModuleComponentRepository.java:378)
                at org.gradle.api.internal.artifacts.ivyservice.ivyresolve.ErrorHandlingModuleComponentRepository$ErrorHandlingModuleComponentRepositoryAccess.resolveComponentMetaData(ErrorHandlingModuleComponentRepository.java:138)
                ... 140 more
        Caused by: org.gradle.internal.resource.transport.http.HttpRequestException: Could not HEAD 'https://jcenter.bintray.com/com/github/johnrengelman/shadow/com.github.johnrengelman.shadow.gradle.plugin/1.2.4/com.github.johnrengelman.shadow.gradle.plugin-1.2.4.pom'.
                at org.gradle.internal.resource.transport.http.HttpClientHelper.performRequest(HttpClientHelper.java:96)
                at org.gradle.internal.resource.transport.http.HttpClientHelper.performRawHead(HttpClientHelper.java:72)
                at org.gradle.internal.resource.transport.http.HttpClientHelper.performHead(HttpClientHelper.java:76)
                at org.gradle.internal.resource.transport.http.HttpResourceAccessor.getMetaData(HttpResourceAccessor.java:65)
                at org.gradle.internal.resource.transfer.DefaultExternalResourceConnector.getMetaData(DefaultExternalResourceConnector.java:63)
                at org.gradle.internal.resource.transfer.AccessorBackedExternalResource.getMetaData(AccessorBackedExternalResource.java:201)
                at org.gradle.internal.resource.BuildOperationFiringExternalResourceDecorator$1.call(BuildOperationFiringExternalResourceDecorator.java:61)
                at org.gradle.internal.resource.BuildOperationFiringExternalResourceDecorator$1.call(BuildOperationFiringExternalResourceDecorator.java:58)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$CallableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:314)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor$CallableBuildOperationWorker.execute(DefaultBuildOperationExecutor.java:304)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.execute(DefaultBuildOperationExecutor.java:174)
                at org.gradle.internal.operations.DefaultBuildOperationExecutor.call(DefaultBuildOperationExecutor.java:100)
                at org.gradle.internal.operations.DelegatingBuildOperationExecutor.call(DelegatingBuildOperationExecutor.java:36)
                at org.gradle.internal.resource.BuildOperationFiringExternalResourceDecorator.getMetaData(BuildOperationFiringExternalResourceDecorator.java:58)
                at org.gradle.internal.resource.transfer.DefaultCacheAwareExternalResourceAccessor$1.create(DefaultCacheAwareExternalResourceAccessor.java:101)
                at org.gradle.internal.resource.transfer.DefaultCacheAwareExternalResourceAccessor$1.create(DefaultCacheAwareExternalResourceAccessor.java:81)
                at org.gradle.cache.internal.ProducerGuard$AdaptiveProducerGuard.guardByKey(ProducerGuard.java:97)
                at org.gradle.internal.resource.transfer.DefaultCacheAwareExternalResourceAccessor.getResource(DefaultCacheAwareExternalResourceAccessor.java:81)
                at org.gradle.api.internal.artifacts.repositories.resolver.DefaultExternalResourceArtifactResolver.downloadByCoords(DefaultExternalResourceArtifactResolver.java:133)
                ... 149 more
        Caused by: java.net.SocketTimeoutException: Read timed out
                at org.apache.http.impl.io.SessionInputBufferImpl.streamRead(SessionInputBufferImpl.java:137)
                at org.apache.http.impl.io.SessionInputBufferImpl.fillBuffer(SessionInputBufferImpl.java:153)
                at org.apache.http.impl.io.SessionInputBufferImpl.readLine(SessionInputBufferImpl.java:282)
                at org.apache.http.impl.conn.DefaultHttpResponseParser.parseHead(DefaultHttpResponseParser.java:138)
                at org.apache.http.impl.conn.DefaultHttpResponseParser.parseHead(DefaultHttpResponseParser.java:56)
                at org.apache.http.impl.io.AbstractMessageParser.parse(AbstractMessageParser.java:259)
                at org.apache.http.impl.DefaultBHttpClientConnection.receiveResponseHeader(DefaultBHttpClientConnection.java:163)
                at org.apache.http.impl.conn.CPoolProxy.receiveResponseHeader(CPoolProxy.java:165)
                at org.apache.http.protocol.HttpRequestExecutor.doReceiveResponse(HttpRequestExecutor.java:273)
                at org.apache.http.protocol.HttpRequestExecutor.execute(HttpRequestExecutor.java:125)
                at org.apache.http.impl.execchain.MainClientExec.execute(MainClientExec.java:272)
                at org.apache.http.impl.execchain.ProtocolExec.execute(ProtocolExec.java:185)
                at org.apache.http.impl.execchain.RetryExec.execute(RetryExec.java:89)
                at org.apache.http.impl.execchain.RedirectExec.execute(RedirectExec.java:111)
                at org.apache.http.impl.client.InternalHttpClient.doExecute(InternalHttpClient.java:185)
                at org.apache.http.impl.client.CloseableHttpClient.execute(CloseableHttpClient.java:83)
                at org.gradle.internal.resource.transport.http.HttpClientHelper.performHttpRequest(HttpClientHelper.java:148)
                at org.gradle.internal.resource.transport.http.HttpClientHelper.performHttpRequest(HttpClientHelper.java:126)
                at org.gradle.internal.resource.transport.http.HttpClientHelper.executeGetOrHead(HttpClientHelper.java:103)
                at org.gradle.internal.resource.transport.http.HttpClientHelper.performRequest(HttpClientHelper.java:94)
                ... 167 more
        
        
        * Get more help at https://help.gradle.org
        
        BUILD FAILED in 30s
        
        
        Minecraft process finished unexpectedly. There was an error with Malmo.
        ```
        </details>
   - **Solution**: Try running the command multiple times, or open a new terminal and rerun the command from there.

