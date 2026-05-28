import os
import argparse
import torch as th
from utils import *
from models.customized_modeling_t5 import CustomizedT5ForConditionalGeneration
from models.adaptive_rmts_t5 import AdaptiveRMTSForConditionalGeneration
from transformers import T5Tokenizer, BartTokenizer, LEDTokenizer, AutoTokenizer
from models.customized_modeling_bart import BartForConditionalGeneration
from models.customized_modeling_pegasus import *
from models.customized_modeling_led import *
import gc
import pickle
import warnings

warnings.filterwarnings("ignore")

def main(args):
    
    set_seed(args)
    
    if not os.path.isdir(f"ckpts_{args.result_path}"):
        os.makedirs(f"ckpts_{args.result_path}")

    args.save_model_path = f"ckpts_{args.result_path}" 
    
    if args.test:
        args.load_checkpoint_path = f"ckpts_{args.result_path}"
    
    if th.cuda.is_available() and args.gpu != -1:
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    
    # ========== 使用本地模型路径加载 tokenizer ==========
    model_name = args.model_name
    local_model_path = f"./models/{model_name}"
    
    if 't5' in model_name or 'flan' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    elif 'bart' in model_name:
        tokenizer = BartTokenizer.from_pretrained(local_model_path)
    elif 'pegasus' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    elif 'led' in model_name:
        tokenizer = LEDTokenizer.from_pretrained(local_model_path)

    if args.data == "asap":        
        if "t5" in args.model_name:
            add_tokens = ["@", "{", "}",'<essay>',"<rationale>","[overall]", "[content]", "[organization]", "[word choice]", "[sentence fluency]", "[conventions]","[prompt adherence]", "[language]", "[narrativity]", "[style]","[voice]", 
                            "overall", "content", "organization", "word choice", "sentence fluency", "conventions", "prompt adherence", "language", "narrativity", "style", "voice"]
        else:
            add_tokens = ["@", "{", "}",'<essay>',"<rationale>","[overall]", "[content]", "[organization]", "[word choice]", "[sentence fluency]", "[conventions]","[prompt adherence]", "[language]", "[narrativity]", "[style]","[voice]"]
        
    else:
        add_tokens = ["@", "{", "}",'<essay>',"<rationale>","[cohesion]", "[syntax]", "[vocabulary]", "[phraseology]", "[grammar]", "[conventions]", "1.0", "1.5", 
                      "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "conventions", "grammar", "vocabulary", "phraseology", "syntax", "cohesion"]
        
    tokenizer.add_tokens(add_tokens)

    best_fold_result_dict = dict()
    best_fold_pred_dict = dict()
    best_fold_true_dict = dict()
    sub_best_fold_result_dict = dict()
    sub_best_fold_pred_dict = dict()
    sub_best_fold_true_dict = dict()
    
    for fold in range(args.folds):
        # ========== 使用本地模型路径加载 model ==========
        if 't5' in args.model_name:
            if args.use_adaptive_rmts:
                model = AdaptiveRMTSForConditionalGeneration.from_pretrained(
                    local_model_path,
                    num_traits=len(args.trait_names),
                    prompt_vocab_size=9 if args.data == "asap" else 1,
                    load_balance_weight=args.load_balance_weight,
                    view_dropout=args.view_dropout,
                    fixed_equal_gate=args.fixed_equal_gate,
                    gate_weights=args.gate_weights if hasattr(args, 'gate_weights') else None,
                )
            else:
                model = CustomizedT5ForConditionalGeneration.from_pretrained(local_model_path)
                model.use_rationale = True
            
        elif 'bart' in args.model_name:
            model = BartForConditionalGeneration.from_pretrained(local_model_path)
            model.model.use_rationale = True

        elif 'pegasus' in args.model_name:
            model = PegasusForConditionalGeneration.from_pretrained(local_model_path)
            model.model.use_rationale = True
            
        elif 'led' in args.model_name:
            model = LEDForConditionalGeneration.from_pretrained(local_model_path)
            model.led.use_rationale = True
        if hasattr(model, "use_rationale"):
            model.rationale_split_index = args.max_essay_length
            model.rationale_length = args.max_rationale_length
        
        model.resize_token_embeddings(len(tokenizer))

        save_model_fold_path = os.path.join(args.save_model_path, str(fold))
        if not os.path.isdir(save_model_fold_path):
            os.makedirs(save_model_fold_path)
        args.save_model_fold_path = save_model_fold_path
        
        if args.data == "asap":
            TRAIN_DATA_PATH = f"./data/essay/fold_{fold}/train.csv"
            DEV_DATA_PATH = f"./data/essay/fold_{fold}/dev.csv"
            TEST_DATA_PATH = f"./data/essay/fold_{fold}/test.csv"
        else:
            TRAIN_DATA_PATH = f"./data/feedback/fold_{fold}/train.csv"
            DEV_DATA_PATH = f"./data/feedback/fold_{fold}/dev.csv"
            TEST_DATA_PATH = f"./data/feedback/fold_{fold}/test.csv"

        train_data = read_data(TRAIN_DATA_PATH)
        dev_data = read_data(DEV_DATA_PATH)
        test_data = read_data(TEST_DATA_PATH)
        
        # 准备评分元数据（用于标准化特征值）
        prepare_score_metadata(train_data, dev_data, test_data, args)
        
        train_dataset = train_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        dev_dataset = dev_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        test_dataset = test_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        
        # 初始化变量，防止未定义错误
        best_result = None
        best_pred_dic = None
        best_true_dic = None
        sub_best_result = None
        sub_best_pred_dic = None
        sub_best_true_dic = None
        
        if not args.test:
            print(f"Model Training Fold : {fold}")
            model = train(model, tokenizer, train_dataset, dev_dataset, args)

            # 检查是否有保存的checkpoint
            if os.path.exists(args.save_model_fold_path):
                checkpoint_files = [f for f in os.listdir(args.save_model_fold_path) if f.startswith("checkpoint")]
            else:
                checkpoint_files = []
            
            if checkpoint_files:
                # 有checkpoint，使用最好的
                for filename in checkpoint_files:
                    if filename.startswith("checkpoint-1"):
                        best_model_path = os.path.join(args.save_model_fold_path, filename)
                        best_checkpoint = th.load(best_model_path)
                        model.load_state_dict(best_checkpoint)
                        best_model = model.to(args.device)
            
                        if args.data == "asap":
                            best_result, best_pred_dic, best_true_dic = asap_test(tokenizer, best_model, test_dataset, args)
                        else:
                            best_result, best_pred_dic, best_true_dic = feedback_test(tokenizer, best_model, test_dataset, args)
                        best_model = best_model.cpu()
                        print(f"Best result from checkpoint: {best_result}")
            
                        del best_model
                        th.cuda.empty_cache()
                        gc.collect()
                        
                        # 找第二个最好的
                        for f in checkpoint_files:
                            if f.startswith("checkpoint-2"):
                                sub_best_model_path = os.path.join(args.save_model_fold_path, f)
                                sub_best_checkpoint = th.load(sub_best_model_path)
                                model.load_state_dict(sub_best_checkpoint)
                                sub_best_model = model.to(args.device)
                                if args.data == "asap":
                                    sub_best_result, sub_best_pred_dic, sub_best_true_dic = asap_test(tokenizer, sub_best_model, test_dataset, args)
                                else:
                                    sub_best_result, sub_best_pred_dic, sub_best_true_dic = feedback_test(tokenizer, sub_best_model, test_dataset, args)
                                sub_best_model = sub_best_model.cpu()
                                
                                del sub_best_model
                                th.cuda.empty_cache()
                                gc.collect()
                                break
            else:
                # 没有checkpoint，直接用训练好的模型测试
                print("No checkpoints found, testing with current model...")
                model = model.to(args.device)
                if args.data == "asap":
                    best_result, best_pred_dic, best_true_dic = asap_test(tokenizer, model, test_dataset, args)
                else:
                    best_result, best_pred_dic, best_true_dic = feedback_test(tokenizer, model, test_dataset, args)
                
                # 也作为sub_best
                sub_best_result, sub_best_pred_dic, sub_best_true_dic = best_result, best_pred_dic, best_true_dic
                
                model = model.cpu()
                th.cuda.empty_cache()
                gc.collect()

        elif args.test:
            print(f"Model Test Fold : {fold}")
            # 检查是否有保存的checkpoint
            if os.path.exists(args.save_model_fold_path):
                checkpoint_files = [f for f in os.listdir(args.save_model_fold_path) if f.startswith("checkpoint")]
            else:
                checkpoint_files = []
                
            for filename in checkpoint_files:
                if filename.startswith("checkpoint-1"):
                    best_model_path = os.path.join(args.save_model_fold_path, filename)
                    best_checkpoint = th.load(best_model_path)
                    model.load_state_dict(best_checkpoint)
                    best_model = model.to(args.device)

                    if args.data == "asap":
                        best_result, best_pred_dic, best_true_dic = asap_test(tokenizer, best_model, test_dataset, args)
                    else:
                        best_result, best_pred_dic, best_true_dic = feedback_test(tokenizer, best_model, test_dataset, args)
                    best_model = best_model.cpu()
        
                    del best_model
                    th.cuda.empty_cache()
                    gc.collect()  
        
                elif filename.startswith("checkpoint-2"):
                    sub_best_model_path = os.path.join(args.save_model_fold_path, filename)
                    sub_best_checkpoint = th.load(sub_best_model_path)
                    model.load_state_dict(sub_best_checkpoint)
                    sub_best_model = model.to(args.device)
                    if args.data == "asap":
                        sub_best_result, sub_best_pred_dic, sub_best_true_dic = asap_test(tokenizer, sub_best_model, test_dataset, args)
                    else:
                        sub_best_result, sub_best_pred_dic, sub_best_true_dic = feedback_test(tokenizer, sub_best_model, test_dataset, args)
                    
                    sub_best_model = sub_best_model.cpu()
                    
                    del sub_best_model
                    th.cuda.empty_cache()
                    gc.collect()  

        # 确保变量有值
        if best_result is None:
            print("Warning: best_result is None, using dummy values")
            best_result = {}
            best_pred_dic = {}
            best_true_dic = {}
        if sub_best_result is None:
            sub_best_result = best_result
            sub_best_pred_dic = best_pred_dic
            sub_best_true_dic = best_true_dic

        best_fold_result_dict[fold] = best_result
        best_fold_pred_dict[fold] = best_pred_dic
        best_fold_true_dict[fold] = best_true_dic
        
        sub_best_fold_result_dict[fold] = sub_best_result
        sub_best_fold_pred_dict[fold] = sub_best_pred_dic
        sub_best_fold_true_dict[fold] = sub_best_true_dic
        
        with open(f"./{args.result_path}/best_result_dict.pkl", "wb") as f:
            pickle.dump(best_fold_result_dict, f)
        with open(f"./{args.result_path}/best_pred_dict.pkl", "wb") as f:
            pickle.dump(best_fold_pred_dict, f)
        with open(f"./{args.result_path}/best_true_dict.pkl", "wb") as f:
            pickle.dump(best_fold_true_dict, f)
        with open(f"./{args.result_path}/sub_best_result_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_result_dict, f)
        with open(f"./{args.result_path}/sub_best_pred_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_pred_dict, f)
        with open(f"./{args.result_path}/sub_best_true_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_true_dict, f)
        
    return best_fold_result_dict, best_fold_pred_dict, best_fold_true_dict, \
        sub_best_fold_result_dict, sub_best_fold_pred_dict, sub_best_fold_true_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Essay Scoring')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='which gpu to use, specify -1 to use CPU')
    parser.add_argument('--train_batch_size', '-trb', type=int, default=4, help='batch_size')
    parser.add_argument('--test_batch_size', '-teb', type=int, default=16, help='test_batch_size')
    parser.add_argument('--seed', '-s', type=int, default=40, help='random seed')
    parser.add_argument('--patience', '-p', type=int, default=5, help='number of patience for early stopping')
    parser.add_argument("--train_epochs", type=int, default=15)
    parser.add_argument("--save_checkpoint_path", type=str, default=None)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--data", type=str, default="asap")
    parser.add_argument("--llm", type=str, default="gpt")
    parser.add_argument('--model_name', '-m', type=str, default='t5-base', help='name of the model')
    parser.add_argument('--learning_rate', '-l', type=float, default=2e-5, help='learning rate')
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--max_essay_length", type=int, default=512)
    parser.add_argument("--max_rationale_length", type=int, default=512)
    parser.add_argument("--use_adaptive_rmts", action="store_true")
    parser.add_argument("--concat_rationales", action="store_true")
    parser.add_argument("--fixed_equal_gate", action="store_true", help="Use fixed equal gate weights (0.5, 0.5)")
    parser.add_argument("--gpt_weight", type=float, default=0.5, help="Weight for GPT rationales (0-1)")
    parser.add_argument("--llama_weight", type=float, default=0.5, help="Weight for Llama rationales (0-1)")
    parser.add_argument("--load_balance_weight", type=float, default=0.1, help="Load balance weight for gate regularization")
    parser.add_argument("--view_dropout", type=float, default=0.1, help="View dropout for regularization")
    
    args = parser.parse_args()
    if args.use_adaptive_rmts and args.concat_rationales:
        raise ValueError("concat_rationales is an ablation for non-adaptive RMTS and cannot be used with use_adaptive_rmts")
    args.trait_names = get_trait_names(args)
    if args.use_adaptive_rmts and args.data == "asap" and args.max_essay_length == 512 and args.max_rationale_length == 512:
        args.max_essay_length = 768
        args.max_rationale_length = 256
    if args.use_adaptive_rmts and args.max_essay_length + args.max_rationale_length != 1024:
        raise ValueError("Adaptive RMTS requires max_essay_length + max_rationale_length == 1024")
    
    # Adaptive RMTS 只使用 load_balance_weight 损失（门控均衡）
    # 其他辅助损失已被移除以简化模型
    if args.use_adaptive_rmts and args.load_balance_weight == 0:
        raise ValueError(
            "load_balance_weight MUST be > 0 (at least 0.05)\n"
            "It prevents the gate from biasing completely towards one view"
        )
    
    # 处理门控权重
    if args.fixed_equal_gate:
        args.gate_weights = (args.gpt_weight, args.llama_weight)
        # 归一化确保和为1
        total = args.gpt_weight + args.llama_weight
        args.gate_weights = (args.gpt_weight / total, args.llama_weight / total)
    
    args.result_path = f"{args.data}"
    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path)
    run_name = f"{args.model_name.replace('/', '_')}_{args.llm}"
    if args.use_adaptive_rmts:
        run_name = f"{run_name}_adaptive"
    if args.concat_rationales:
        run_name = f"{run_name}_concat_rationales"
    if args.fixed_equal_gate:
        run_name = f"{run_name}_fixed_gate_{args.gpt_weight:.1f}_{args.llama_weight:.1f}"
    args.result_path = os.path.join(args.result_path, run_name)
    
    best_fold_result_dict, best_fold_pred_dict, best_fold_true_dict, \
        sub_best_fold_result_dict, sub_best_fold_pred_dict, sub_best_fold_true_dict = main(args)

    with open(f"./{args.result_path}/final_best_result_dict.pkl", "wb") as f:
        pickle.dump(best_fold_result_dict, f)
    with open(f"./{args.result_path}/final_best_pred_dict.pkl", "wb") as f:
        pickle.dump(best_fold_pred_dict, f)
    with open(f"./{args.result_path}/final_best_true_dict.pkl", "wb") as f:
        pickle.dump(best_fold_true_dict, f)
    with open(f"./{args.result_path}/final_sub_best_result_dict.pkl", "wb") as f:
        pickle.dump(sub_best_fold_result_dict, f)
    with open(f"./{args.result_path}/final_sub_best_pred_dict.pkl", "wb") as f:
        pickle.dump(sub_best_fold_pred_dict, f)
    with open(f"./{args.result_path}/final_sub_best_true_dict.pkl", "wb") as f:
        pickle.dump(sub_best_fold_true_dict, f)

