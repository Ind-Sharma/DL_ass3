import wandb

# Put your key here
WANDB_API_KEY = "your_key_here"
PROJECT_NAME = "da6401-assignment3"


def main():
    if WANDB_API_KEY == "PASTE_YOUR_WANDB_API_KEY_HERE":
        raise RuntimeError("Set WANDB_API_KEY in login_wandb.py first.")

    wandb.login(key=WANDB_API_KEY, relogin=True)
    run = wandb.init(project=PROJECT_NAME, job_type="login-check", reinit=True)
    run.log({"wandb_login_ok": 1})
    run.finish()
    print("W&B login successful.")


if __name__ == "__main__":
    main()
