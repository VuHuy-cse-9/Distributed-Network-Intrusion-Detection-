from trainer import Trainer
from global_trainer import GlobalTrainer
from options import SystemOptions

options = SystemOptions()
opts = options.parse()


if __name__ == "__main__":
    if opts.islocal == "True":
        trainer = Trainer(opts)
        trainer.summary_dataset()
        trainer.train()
        trainer.eval()
        trainer.save()
    else:
        print(">> Initialize global model ...")
        trainer = GlobalTrainer(opts)
        print(">> Print Summary ...")
        trainer.summary_dataset()
        print(">> Local training ...")
        trainer.train_local()
        print(">> Send model ...")
        trainer._send_model()
        print(">> Wait for receiving model ...")
        trainer.receive_model()
        print(">> Global training ...")
        trainer.train_global()
        print(">> Evaluating ...")
        trainer.eval_global()
        trainer.save()