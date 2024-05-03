import os
import numpy as np
from semanticodec import SemantiCodec
import soundfile as sf

testaudiopath = "/mnt/bn/lqhaoheliu/hhl_script2/2024/SemanticCodec/build_evaluation_set/evaluationset_16k"
checkpoint_path = "/mnt/bn/hlhaoheliu2/semanticodec_log/43_search_best/2024_04_11_lstm_bidirectional_512_rand_centroid/checkpoints"

checkpoint_path_list = [os.path.join(checkpoint_path, x) for x in os.listdir(checkpoint_path) if ".ckpt" in x]
np.random.shuffle(checkpoint_path_list)
semanticodec = SemantiCodec(token_rate=50, vocab_size=32768)

for checkpoint_path in checkpoint_path_list:
    output_save_path = "/mnt/bn/lqhaoheliu/project/SemantiCodec/final_sc_512/output_%s" % os.path.basename(checkpoint_path).replace(".ckpt","")
    for file in os.listdir(testaudiopath):
        filepath = os.path.join(testaudiopath, file)
        tokens = semanticodec.encode(filepath)
        waveform = semanticodec.decode(tokens)
        sf.write(os.path.join(output_save_path, file), waveform[0,0], 16000)