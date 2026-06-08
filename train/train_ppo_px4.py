"""Train PPO online against PX4 SITL and the Gazebo wire scene."""

from pathlib import Path
import argparse
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_LOG_DIR = PROJECT_ROOT / "logs"
os.environ.setdefault("ROS_LOG_DIR", str(DEFAULT_RUN_LOG_DIR / "ros"))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RUN_LOG_DIR / "matplotlib"))
Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from uav_px4_rl.env import Px4GazeboWireEnv
from uav_px4_rl.px4_backend import Px4RosBackend


DEFAULT_LIDAR_TOPIC = "/x500/lidar/points"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument(
        "--scenario",
        choices=["fixed", "random"],
        default="random",
        help="'random' is the 3B-lite curriculum; 'fixed' is the pipeline check.",
    ) #场景模式 fixed 固定场景 random 随机场景
    parser.add_argument("--seed", type=int, default=0)  #
    parser.add_argument("--num-wires", type=int, default=3)
    parser.add_argument("--perception", default="lidar", choices=["lidar", "empty", "none"])
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--model-name", default="ppo_px4_3d_lidar_multiwire")
    parser.add_argument("--setup-timeout", type=float, default=60.0)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-freq", type=int, default=2_000)
    parser.add_argument(
        "--resume-model",
        type=Path,
        help="Continue training from an existing PPO .zip model.",
    )
    simulation_mode = parser.add_mutually_exclusive_group()
    simulation_mode.add_argument(
        "--realtime",
        dest="synchronous",
        action="store_false",
        help="Run the PX4/Gazebo loop in real time (default).",
    )
    simulation_mode.add_argument(
        "--synchronous",
        dest="synchronous",
        action="store_true",
        help="Pause/step Gazebo for lockstep diagnostics; not recommended for long training.",
    )
    parser.set_defaults(synchronous=False)
    return parser.parse_args() #返回解析后的命令行参数


def main():
    args = parse_args() 
    model_dir = PROJECT_ROOT / "models" #模型保存目录
    log_dir = PROJECT_ROOT / "logs" #日志保存目录
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        backend = Px4RosBackend(
            synchronous=args.synchronous,
            lidar_topic=args.lidar_topic,
            setup_timeout=args.setup_timeout,
        ) #创建PX4ROS后端
    except TimeoutError as exc:
        print(
            "PX4/Gazebo stack is not ready for training. Start terminal A with "
            "`GUI=false bash tools/start_stack.sh`, wait for PX4 startup, then "
            "verify `/fmu/out/vehicle_odometry` and `/x500/lidar/points` before "
            "running PPO training.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    env = Px4GazeboWireEnv( #创建PX4GazeboWireEnv环境
        backend=backend,
        scenario_mode=args.scenario, #场景模式
        seed=args.seed, #随机种子
        num_wires=args.num_wires,
        perception_mode=args.perception,
    )
    env = Monitor(env, filename=str(log_dir / f"{args.model_name}_monitor.csv"))

    model = None
    try:
        checkpoint = CheckpointCallback( #创建检查点回调
            save_freq=args.checkpoint_freq, #保存频率
            save_path=str(model_dir / "checkpoints"), #保存路径
            name_prefix=args.model_name, #模型名称
        )
        if args.resume_model is not None:
            model = PPO.load(
                str(args.resume_model),
                env=env,
                tensorboard_log=str(log_dir / "tensorboard"),
            )
            print(f"Continuing PPO training from {args.resume_model}.")
        else:
            model = PPO(
                "MlpPolicy", #MLP策略
                env,
                learning_rate=3e-4,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1, #详细程度
                seed=args.seed, #随机种子
                tensorboard_log=str(log_dir / "tensorboard"), #TensorBoard日志路径
            )
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint,
            progress_bar=True,
            reset_num_timesteps=args.resume_model is None,
        )
        output_path = model_dir / args.model_name #模型保存路径
        model.save(str(output_path)) #保存模型
        print(f"Online PX4/Gazebo PPO model saved to {output_path}.zip") #打印模型保存路径
    except (Exception, KeyboardInterrupt):
        if model is not None:
            interrupted_path = model_dir / f"{args.model_name}_interrupted"
            model.save(str(interrupted_path))
            print(
                f"Training stopped before completion; latest model saved to "
                f"{interrupted_path}.zip",
                file=sys.stderr,
            )
        raise
    finally: #最终关闭环境
        env.close() #关闭环境


if __name__ == "__main__":
    main()
