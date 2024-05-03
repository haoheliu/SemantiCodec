import os
import numpy as np
from semanticodec import SemantiCodec
import soundfile as sf

testaudiopath = "/mnt/bn/lqhaoheliu/hhl_script2/2024/SemanticCodec/build_evaluation_set/evaluationset_16k"
MAX=10

############## 512
checkpoint_path = "/mnt/bn/hlhaoheliu2/semanticodec_log/43_search_best/2024_04_11_lstm_bidirectional_512_rand_centroid/checkpoints"
checkpoint_path_list = [os.path.join(checkpoint_path, x) for x in os.listdir(checkpoint_path) if ".ckpt" in x]
np.random.shuffle(checkpoint_path_list)
checkpoint_path_list = checkpoint_path_list[:MAX]
semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=32768)

for checkpoint_path in checkpoint_path_list:
    output_save_path = "/mnt/bn/lqhaoheliu/project/SemantiCodec/final_sc_512/output_%s" % os.path.basename(checkpoint_path).replace(".ckpt","")
    if os.path.exists(output_save_path):
        continue
    os.makedirs(output_save_path, exist_ok=True)
    print("output_save_path: ", output_save_path)

    for file in os.listdir(testaudiopath):
        filepath = os.path.join(testaudiopath, file)
        tokens = semanticodec.encode(filepath)
        print("512", tokens.shape)
        waveform = semanticodec.decode(tokens)
        sf.write(os.path.join(output_save_path, file), waveform[0,0], 16000)

############## 256
checkpoint_path = "/mnt/bn/hlhaoheliu2/semanticodec_log/43_search_best/2024_04_11_lstm_bidirectional_256_rand_centroid/checkpoints"
checkpoint_path_list = [os.path.join(checkpoint_path, x) for x in os.listdir(checkpoint_path) if ".ckpt" in x]
np.random.shuffle(checkpoint_path_list)
checkpoint_path_list = checkpoint_path_list[:MAX]
semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=32768)

for checkpoint_path in checkpoint_path_list:
    output_save_path = "/mnt/bn/lqhaoheliu/project/SemantiCodec/final_sc_256/output_%s" % os.path.basename(checkpoint_path).replace(".ckpt","")
    if os.path.exists(output_save_path):
        continue
    os.makedirs(output_save_path, exist_ok=True)
    print("output_save_path: ", output_save_path)

    for file in os.listdir(testaudiopath):
        filepath = os.path.join(testaudiopath, file)
        tokens = semanticodec.encode(filepath)
        print(256, tokens.shape)
        waveform = semanticodec.decode(tokens)
        sf.write(os.path.join(output_save_path, file), waveform[0,0], 16000)

############## 128
checkpoint_path = "/mnt/bn/hlhaoheliu2/semanticodec_log/43_search_best/2024_04_11_lstm_bidirectional_128_rand_centroid/checkpoints"
checkpoint_path_list = [os.path.join(checkpoint_path, x) for x in os.listdir(checkpoint_path) if ".ckpt" in x]
np.random.shuffle(checkpoint_path_list)
checkpoint_path_list = checkpoint_path_list[:MAX]
semanticodec = SemantiCodec(token_rate=25, semantic_vocab_size=32768)

for checkpoint_path in checkpoint_path_list:
    output_save_path = "/mnt/bn/lqhaoheliu/project/SemantiCodec/final_sc_128/output_%s" % os.path.basename(checkpoint_path).replace(".ckpt","")
    if os.path.exists(output_save_path):
        continue
    os.makedirs(output_save_path, exist_ok=True)
    print("output_save_path: ", output_save_path)

    for file in os.listdir(testaudiopath):
        filepath = os.path.join(testaudiopath, file)
        tokens = semanticodec.encode(filepath)
        print(128, tokens.shape)
        waveform = semanticodec.decode(tokens)
        sf.write(os.path.join(output_save_path, file), waveform[0,0], 16000)