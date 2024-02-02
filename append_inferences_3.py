import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import csv
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15-maxr_5/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full/1e-05_adam_64_13500.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    data=pd.read_csv("scripts/interactive/train_with_test.tsv",sep="\t",header=0)
    

    data["new_sentences"]=" "
    data["new_ind"]=0
    extensions={
        "Desires" : "desires",
        "CapableOf":"is capable of",
        "HasProperty":"has property",
        "AtLocation":"is at location",
        "UsedFor":"is used for",
        "MadeOf":"is made of"
    }
    count=0
    #count2=0
    for index,row in data.iterrows():
        
        input_sentence=row["sentence"]
        v_ind=int(row["v_index"])
        #print(input_sentence)
        #print(len(input_sentence.split()))
        #print(v_ind)
        if len(input_sentence.split())-1<v_ind:
            data.at[index,"new_sentences"]=input_sentence
            data.at[index,"new_ind"]=v_ind
            print("Inside if")
            #print("Inside")
            continue
        input_event=input_sentence.split()[v_ind]
        input_event2=row["new_column"]
        if isinstance(input_event2,float):
             print("Inside if")
             data.at[index,"new_sentences"]=input_sentence
             data.at[index,"new_ind"]=v_ind
             continue
        #print(not (isinstance(input_event2,str) or isinstance(input_event2,float)))
        #print(type(input_event2))
        #print("continuing otside")
        #print("Input event 1: ",input_event)
        #print("Input event 2: ",input_event2)
        #print("Input sentence : ",input_sentence)
        sentences=[]
        relations=["HasProperty","AtLocation","UsedFor"]
        for relation in relations:
            sampling_algorithm = "beam-1"
            sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
            outputs = interactive.get_conceptnet_sequence(
                input_event, model, sampler, data_loader, text_encoder, relation)
            sentence=outputs[relation]["e1"]+" "+extensions[relation]+" "+outputs[relation]["beams"][0]+"."
            sentences.append(sentence)
        relations2=["Desires","AtLocation","UsedFor"]
        for relation in relations2:
            sampling_algorithm = "beam-1"
            sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
            outputs = interactive.get_conceptnet_sequence(
                input_event2, model, sampler, data_loader, text_encoder, relation)
            sentence=outputs[relation]["e1"]+" "+extensions[relation]+" "+outputs[relation]["beams"][0]+"."
            sentences.append(sentence)
        sentences_copy=sentences[:]
        sentences.append(row["sentence"])
        sentence_str=" ".join(e for e in sentences)
        sent_str_2=" ".join(e for e in sentences_copy)
        lis=sentence_str.split()
        lis_2=sent_str_2.split()
        new_ind=len(lis_2)+row["v_index"]
        data.at[index,"new_sentences"]=sentence_str
        data.at[index,"new_ind"]=new_ind
        count+=1
        #print(count)
    data.to_csv("Final_test_data.tsv",sep="\t",index=False)


    
