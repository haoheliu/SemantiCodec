import os
import json

import torch
import numpy as np

import hifigan


def get_available_checkpoint_keys(model, ckpt):
    print("==> Attemp to reload from %s" % ckpt)
    state_dict = torch.load(ckpt)["state_dict"]
    current_state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if (
            k in current_state_dict.keys()
            and current_state_dict[k].size() == state_dict[k].size()
        ):
            new_state_dict[k] = state_dict[k]
        else:
            print("==> WARNING: Skipping %s" % k)
    print(
        "%s out of %s keys are matched"
        % (len(new_state_dict.keys()), len(state_dict.keys()))
    )
    return new_state_dict

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def torch_version_orig_mod_remove(state_dict):
    new_state_dict = {}
    new_state_dict["generator"] = {}
    for key in state_dict["generator"].keys():
        if("_orig_mod." in key):
            new_state_dict["generator"][key.replace("_orig_mod.","")] = state_dict["generator"][key]
        else:
            new_state_dict["generator"][key] = state_dict["generator"][key]
    return new_state_dict

def get_vocoder(config, device, mel_bins):
    name = "HiFi-GAN"
    speaker = ""
    ROOT = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/src"
    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        if mel_bins == 64:
            with open(os.path.join(ROOT, "hifigan/config_16k_64.json"), "r") as f:
                config = json.load(f)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator_old(config)
            # print("Load hifigan/g_01080000")
            # ckpt = torch.load(os.path.join(ROOT, "hifigan/g_01080000"))
            print("Load hifigan/g_00660000")
            ckpt = torch.load(os.path.join(ROOT, "hifigan/g_00660000"))
            ckpt = torch_version_orig_mod_remove(ckpt)
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
        elif mel_bins == 128:
            with open("hifigan/config_16k_128.json", "r") as f:
                config = json.load(f)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator_old(config)
            print("Load hifigan/g_01440000")
            ckpt = torch.load(os.path.join(ROOT, "hifigan/g_01440000"))
            ckpt = torch_version_orig_mod_remove(ckpt)
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
        # elif mel_bins == 256:
        #     with open(os.path.join(ROOT, "hifigan/config_hifigan_32k_256.json"), "r") as f:
        #         config = json.load(f)
        #     config = hifigan.AttrDict(config)
        #     vocoder = hifigan.Generator(config)
        #     print("Load hifigan/g_01910000")
        #     ckpt = torch.load(os.path.join(ROOT, "hifigan/g_01910000"))
        #     ckpt = torch_version_orig_mod_remove(ckpt)
        #     vocoder.load_state_dict(ckpt["generator"])
        #     vocoder.eval()
        #     vocoder.remove_weight_norm()
        #     vocoder.to(device)
        # elif mel_bins == 256:
        #     with open(os.path.join(ROOT, "hifigan/config_48k_1024.json"), "r") as f:
        #         config = json.load(f)
        #     config = hifigan.AttrDict(config)
        #     vocoder = hifigan.Generator(config)
        #     print("Load hifigan/g_00650000")
        #     ckpt = torch.load(os.path.join(ROOT, "hifigan/g_00650000"))
        #     ckpt = torch_version_orig_mod_remove(ckpt)
        #     vocoder.load_state_dict(ckpt["generator"])
        #     vocoder.eval()
        #     vocoder.remove_weight_norm()
        #     vocoder.to(device)
        elif mel_bins == 256:
            with open(os.path.join(ROOT, "hifigan/config_48k_1536.json"), "r") as f:
                config = json.load(f)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator(config)
            print("Load hifigan/g_00280000")
            ckpt = torch.load(os.path.join(ROOT, "hifigan/g_00280000"))
            ckpt = torch_version_orig_mod_remove(ckpt)
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    # wavs = [wav for wav in wavs]

    # for i in range(len(mels)):
    #     if lengths is not None:
    #         wavs[i] = wavs[i][: lengths[i]]

    return wavs
