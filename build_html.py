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
    "musdb_91other.wav",
    "audioset_Yycs92N_rph8.wav",
    "musdb_6vocals.wav",
    "audioset_YzdRlVEP4DJo.wav",
    "musdb_159vocals.wav",
    "audioset_YYqc71wg948Y.wav",
    "libritts_1089_134686_000009_000006.wav",
    "musdb_243drums.wav",
    "audioset_Ywy5edFMFcyM.wav",
    "libritts_7021_85628_000029_000000.wav",
    "libritts_121_127105_000044_000001.wav",
    "audioset_YyCvrvgukN4U.wav",
    "audioset_Yw_z9oSn-eIM.wav",
    "libritts_2300_131720_000041_000001.wav",
    "audioset_YZJU9afqgUV0.wav",
    "musdb_247other.wav",
    "audioset_YzhrRL8yzTEs.wav",
    "musdb_194mixture.wav"
]
count = 0
for file in filelist:
    if file not in target_filelist:
        continue
    count += 1
    html_string += "<tr>"
    html_string += f'<td>{count}</td>'
    for column in table_column:
        html_string += f'<td><audio controls="controls"><source src="audio/{column}/{file}" autoplay />Your browser does not support the audio element.</audio></td>'
    html_string += "</tr>"
print(html_string)
