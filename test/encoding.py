from semanticodec import SemantiCodec
import soundfile as sf

###############Choose one of the following######################
semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=32768) # 1.40 kbps
semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=32768) # 0.70 kbps
semanticodec = SemantiCodec(token_rate=25, semantic_vocab_size=32768) # 0.35 kbps

semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=16384) # 1.35 kbps
semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=16384) # 0.68 kbps
semanticodec = SemantiCodec(token_rate=25, semantic_vocab_size=16384) # 0.34 kbps

semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=8192) # 1.30 kbps
semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=8192) # 0.65 kbps
semanticodec = SemantiCodec(token_rate=25, semantic_vocab_size=8192) # 0.33 kbps

semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=4096) # 1.25 kbps
semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=4096) # 0.63 kbps
semanticodec = SemantiCodec(token_rate=25, semantic_vocab_size=4096) # 0.31 kbps
#####################################

filepath = "test.wav"

tokens = semanticodec.encode(filepath)
waveform = semanticodec.decode(tokens)

sf.write("output.wav", waveform[0,0], 16000)
