import os
import argparse

class SystemOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="DIDS options")
        
        #PATH:
        self.parser.add_argument('--path_train',
                                type=str,
                                help="Path to csv trained file",
                                default="dataset/kddcup1999_10percent.csv")
        self.parser.add_argument('--path_test',
                                type=str,
                                help="Path to csv tested file",
                                default="dataset/test.csv")
        self.parser.add_argument('--path_save_dir',
                                type=str,
                                help="Path to directory save model parameters",
                                default="checkpoint/")
        self.parser.add_argument('--path_result',
                                type=str,
                                help="Path to directory save model parameters",
                                default="results/")
        
        #TRAINING OPTIONS
        self.parser.add_argument('--islocal',
                                 type=str,
                                 help="if islocal true, local train is conducted",
                                 default="True",
                                 choices=["True", "False"])
        
        self.parser.add_argument('--train_mode',
                                type=str,
                                default="full",
                                help='Mode of data to train model',
                                choices=["full", "part", "rand"])
        
        self.parser.add_argument('--cate2cont', 
                            type=str, 
                            default="integer", 
                            help='Method to convert categorize data to continuous data',
                            choices=["integer", "remove"])
        
        self.parser.add_argument('--log_transform', 
                            type=str, 
                            default="True", 
                            help='If True, all data is performed log transformation',
                            choices=["True", "False"]
                            )
        
        #GMMs hyperameter:
        self.parser.add_argument('--default_std',
                                type=float,
                                help="Default standard deviation when reset Gaussian components, equation (15)",
                                default=1.0)
        
        self.parser.add_argument('--n_components',
                                 type=int,
                                 help="Number of components of each GMMs",
                                 default=4)
        
        self.parser.add_argument('--T',
                                type=float,
                                help="Confidence limits required for a GMM's component to select that sample, \
                                     equation (5)",
                                default=0.5)
        
        self.parser.add_argument('--gamma',
                                 type=int,
                                 help="Gamma controls the number of weak classifiers \
                                        that are chosen for further updating, equation (25)",
                                 default=10)
        
        self.parser.add_argument('--r',
                                type=float,
                                help="r controls the tradeoff between detection rate (approach 1) \
                                        and false alarm rate (approach 0)",
                                default=0.5)
        
        self.parser.add_argument('--p',
                                type=float,
                                help="p controls the contribution between historical false classification rate ε (approach 0)\
                                        and current sample false classification rate (approach 1)",
                                default=0.1)
        
        self.parser.add_argument('--beta',
                                type=float,
                                help="beta controls the contribution between current ensemble classifier (approach 1)\
                                    and last ensemble classifiers to ensemble weights alpha, equation (29)",
                                default=0.8)
        #Local hyperameter:
        self.parser.add_argument('--P',
                                type=int,
                                help="Maximum number of iteration for updating GMMs, equation (25)",
                                default=8)
        
        #Global hyperamter:
        self.parser.add_argument('--kafka_server',
                                type=str,
                                help="url to kafka server",
                                default="localhost:9092")
        
        self.parser.add_argument('--N_particles', 
                                type=int,
                                help="Number of particles use to train in Global algorithm.",
                                default=3)
        self.parser.add_argument('--N_nodes',
                                 type=int,
                                 help="Number of nodes involve to train in Globa Algorithm",
                                 default=6,
                                 choices=[2, 6])
        self.parser.add_argument('--N_send_samples',
                                 type=int,
                                 help="Number of local sample Node share for others",
                                 default=500)
        
        self.parser.add_argument('--tau',
                                type=float,
                                help="tau control the trade off of finess value between detection rate\
                                        and number of models involves at the particles, equation (34)",
                                default=0.75)
        
        self.parser.add_argument('--u1',
                                type=float,
                                help="relatively independent random values in the range [0, 1], equation (37)",
                                default=0.2)
        
        self.parser.add_argument('--u2',
                                type=float,
                                help="relatively independent random values in the range [0, 1], equation (37)",
                                default=0.3)
        
        self.parser.add_argument('--c1',
                                type=float,
                                help="acceleration constants called the learning rates for states approach best state, equation (37)",
                                default=0.1)
        
        self.parser.add_argument('--c2',
                                type=float,
                                help="acceleration constants called the learning rates for states approach global state, equation (37)",
                                default=0.1)
        
        self.parser.add_argument('--inertia_weight_mode',
                                type=str,
                                help="mode for estimate inertia weight, that is negatively correlated with the number of iterations. \
                                        vary mode for range [1.4 - 0.4], fix for 1.4",
                                default="fix",
                                choices=["fix", "vary"])
        self.parser.add_argument('--N_iter',
                                 type=int,
                                 help="Number of iteration for updating global algorithm",
                                 default=20)
        self.parser.add_argument('--V_max',
                                 type=float,
                                 help="Above aboundary for updating's Velocity ",
                                 default=2.0)
        self.parser.add_argument('--N_global_train',
                                 type=int,
                                 help="Number of train sample for global training",
                                 default=200)
        self.parser.add_argument('--N_global_test',
                                 type=int,
                                 help="Number of test sample for global training",
                                 default=100)

        #OPTIMIZATION OPTIONS
        
        #ABLATION OPTIONS
        
        
        #SYSTEM OPTIONS
        self.parser.add_argument('--nodeid', 
                                 type=int,
                                 default=1,  
                                 help='Node id for save checkpoint later')

        #LOADING OPTIONS
        
        
        #LOGGING OPTIONS
        
        
        #EVALUATION OPTIONS
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options