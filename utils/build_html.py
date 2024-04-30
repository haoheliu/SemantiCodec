import os

table_column = ["groundtruth","encodec_24k_3kbps","hifi-codec-2kbps","encodec_24k_1_5kbps","descript_codec_1_41kbps","semanticodec_1_43_kbps","descript_codec_0_71kbps","semanticodec_0_71_kbps","descript_codec_0_47kbps","semanticodec_0_35_kbps"]

root_dir = "/Users/haoheliu/Project/codec-main/audio"
filelist = os.listdir(os.path.join(root_dir, "groundtruth"))

# <tr> ... </tr>

# <td><audio controls="controls">
#     <source
#     src="audio/semanticodec_0_35_kbps/audioset_Y-xfgovG6-KU.wav"
#     autoplay />Your browser does not support the audio element.
# </audio></td>
html_string = ""

# musdb_91other.wav
# audioset_Yycs92N_rph8.wav
# musdb_6vocals.wav
# audioset_YzdRlVEP4DJo.wav
# musdb_159vocals.wav
# audioset_YYqc71wg948Y.wav
# libritts_1089_134686_000009_000006.wav
# musdb_243drums.wav
# audioset_Ywy5edFMFcyM.wav
# libritts_7021_85628_000029_000000.wav
# libritts_121_127105_000044_000001.wav
# audioset_YyCvrvgukN4U.wav
# audioset_Yw_z9oSn-eIM.wav
# libritts_2300_131720_000041_000001.wav
# audioset_YZJU9afqgUV0.wav
# musdb_247other.wav
# audioset_YzhrRL8yzTEs.wav
# musdb_194mixture.wav

target_filelist = [
    # "audioset_Yycs92N_rph8.wav",
    # "audioset_YzhrRL8yzTEs.wav",
    # "audioset_Y-xfgovG6-KU.wav",
    # "audioset_YzdRlVEP4DJo.wav",
    # "audioset_YYqc71wg948Y.wav",
    # "audioset_YX7qFgrAl3OU.wav",
    # "audioset_Ywy5edFMFcyM.wav",
    # "audioset_YxLZp-3_71Vs.wav",
    # "audioset_YyCvrvgukN4U.wav",
    # "audioset_Yw_z9oSn-eIM.wav",
    # "audioset_YyrwqjyLeUDY.wav",
    # "audioset_Yyc4mmvlO39M",
    # "audioset_YZJU9afqgUV0.wav",


    # "libritts_8555_284449_000009_000000.wav",
    # "libritts_7176_92135_000069_000002.wav",
    # "libritts_5639_40744_000012_000003.wav"
    # "libritts_7021_85628_000029_000000.wav",
    # "libritts_121_127105_000044_000001.wav",
    # "libritts_237_134493_000006_000002.wav",
    # "libritts_2300_131720_000041_000001.wav",
    # "libritts_1089_134686_000009_000006.wav",
    # "libritts_4992_23283_000047_000002.wav",

    "musdb_247other.wav",
    "musdb_33other.wav",
    "musdb_109vocals.wav",
    "musdb_6vocals.wav",
    "musdb_159vocals.wav",
    "musdb_243drums.wav",
    "musdb_123mixture.wav",
    "musdb_161drums.wav",
    "musdb_193drums.wav",
    "musdb_91other.wav",
    "musdb_194mixture.wav"
]
count = 0
for file in target_filelist:
    count += 1
    html_string += "<tr>"
    html_string += f'<td>{count}</td>'
    for column in table_column:
        html_string += f'<td><audio controls="controls"><source src="audio/{column}/{file}" autoplay />Your browser does not support the audio element.</audio></td>'
    html_string += "</tr>"
print(html_string)
