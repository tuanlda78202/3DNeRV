import torch
import imageio
import os
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.utils.data
from copy import deepcopy
from dahuffman import HuffmanCodec
from torchvision.utils import save_image


def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint_dir = "/home/tuanlda78202/3ai24/ckpt/"

    arch = type(model).__name__

    state = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }

    filename = str(checkpoint_dir + "checkpoint-epoch{}.pth".format(epoch))

    torch.save(state, filename)

    print("Saving checkpoint: {} ...".format(filename))


def resume_checkpoint(model, optimizer, resume_path):
    resume_path = str(resume_path)

    print("Loading checkpoint: {} ...".format(resume_path))

    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint["epoch"] + 1

    # load architecture params from checkpoint.
    if checkpoint["arch"] != model:
        print(
            "Warning: Architecture configuration given in config file is different from that of "
            "checkpoint. This may yield an exception while state_dict is being loaded."
        )
    else:
        model.load_state_dict(checkpoint["state_dict"])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint["optimizer"] != optimizer:
        print(
            "Warning: Optimizer type given in config file is different from that of checkpoint. "
            "Optimizer parameters not being resumed."
        )
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])

    print("Checkpoint loaded. Resume training from epoch {}".format(start_epoch))


"""
# Evaluation training
@torch.no_grad()
def evaluated(
    model, full_dataloader, local_rank, args, dump_vis=False, huffman_coding=False
):
    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]
    for model_ind, cur_model in enumerate(model_list):
        time_list = []
        cur_model.eval()
        device = next(cur_model.parameters()).device
        if dump_vis:
            visual_dir = f"{args.outf}/visualize_model" + (
                "_quant" if model_ind else "_orig"
            )
            print(f"Saving predictions to {visual_dir}...")
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)

        for i, sample in enumerate(full_dataloader):
            img_data, norm_idx, img_idx = (
                data_to_gpu(sample["img"], device),
                data_to_gpu(sample["norm_idx"], device),
                data_to_gpu(sample["idx"], device),
            )
            if i > 10 and args.debug:
                break
            img_data, img_gt, inpaint_mask = args.transform_func(img_data)
            cur_input = norm_idx if "pe" in args.embed else img_data
            img_out, embed_list, dec_time = cur_model(
                cur_input, dequant_vid_embed[i] if model_ind else None
            )
            if model_ind == 0:
                img_embed_list.append(embed_list[0])

            # collect decoding fps
            time_list.append(dec_time)
            if args.eval_fps:
                time_list.pop()
                for _ in range(100):
                    img_out, embed_list, dec_time = cur_model(cur_input, embed_list[0])
                    time_list.append(dec_time)

            # compute psnr and ms-ssim
            pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch(
                [img_out], img_gt
            )
            for metric_idx, cur_v in enumerate([pred_psnr, pred_ssim]):
                for batch_i, cur_img_idx in enumerate(img_idx):
                    metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                    metric_list[metric_idx_start + metric_idx + 4 * model_ind].append(
                        cur_v[:, batch_i]
                    )

            # dump predictions
            if dump_vis:
                for batch_ind, cur_img_idx in enumerate(img_idx):
                    full_ind = i * args.batchSize + batch_ind
                    dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                    temp_psnr_list = ",".join(
                        [str(round(x[batch_ind].item(), 2)) for x in pred_psnr]
                    )
                    concat_img = torch.cat(dump_img_list, dim=2)  # img_out[batch_ind],
                    save_image(
                        concat_img,
                        f"{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png",
                    )

            # print eval results and add to log txt
            if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
                avg_time = sum(time_list) / len(time_list)
                fps = args.batchSize / avg_time
                print_str = "[{}] Rank:{}, Eval at Step [{}/{}] , FPS {}, ".format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                    local_rank,
                    i + 1,
                    len(full_dataloader),
                    round(fps, 1),
                )
                metric_name = ("quant" if model_ind else "pred") + "_seen_psnr"
                for v_name, v_list in zip(args.metric_names, metric_list):
                    if metric_name in v_name:
                        cur_value = (
                            torch.stack(v_list, dim=-1).mean(-1)
                            if len(v_list)
                            else torch.zeros(1)
                        )
                        print_str += f"{v_name}: {RoundTensor(cur_value, 2)} | "
                if local_rank in [0, None]:
                    print(print_str, flush=True)
                    with open("{}/rank0.txt".format(args.outf), "a") as f:
                        f.write(print_str + "\n")

        # embedding quantization
        if model_ind == 0:
            vid_embed = torch.cat(img_embed_list, 0)
            quant_embed, dequant_emved = quant_tensor(vid_embed, args.quant_embed_bit)
            dequant_vid_embed = dequant_emved.split(args.batchSize, dim=0)

        # Collect results from
        results_list = [
            torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1)
            for v_list in metric_list
        ]
        args.fps = fps
        h, w = img_data.shape[-2:]
        cur_model.train()
        if args.distributed and args.ngpus_per_node > 1:
            for cur_v in results_list:
                cur_v = all_reduce([cur_v.to(local_rank)])

        # Dump predictions and concat into videos
        if dump_vis and args.dump_videos:
            gif_file = os.path.join(
                args.outf, "gt_pred" + ("_quant.gif" if model_ind else ".gif")
            )
            with imageio.get_writer(gif_file, mode="I") as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.v2.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)
            # optimize(gif_file)

    # dump quantized checkpoint, and decoder
    if local_rank in [0, None] and quant_ckt != None:
        quant_vid = {"embed": quant_embed, "model": quant_ckt}
        torch.save(quant_vid, f"{args.outf}/quant_vid.pth")
        torch.jit.save(
            torch.jit.trace(HNeRVDecoder(model), (vid_embed[:2])),
            f"{args.outf}/img_decoder.pth",
        )
        # huffman coding
        if huffman_coding:
            quant_v_list = quant_embed["quant"].flatten().tolist()
            tmin_scale_len = (
                quant_embed["min"].nelement() + quant_embed["scale"].nelement()
            )
            for k, layer_wt in quant_ckt.items():
                quant_v_list.extend(layer_wt["quant"].flatten().tolist())
                tmin_scale_len += (
                    layer_wt["min"].nelement() + layer_wt["scale"].nelement()
                )

            # get the element name and its frequency
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]

            # total bits for quantized embed + model weights
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            args.bits_per_param = total_bits / len(quant_v_list)

            # including the overhead for min and scale storage,
            total_bits += tmin_scale_len * 16  # (16bits for float16)
            args.full_bits_per_param = total_bits / len(quant_v_list)

            # bits per pixel
            args.total_bpp = total_bits / args.final_size / args.full_data_length
            print(
                f"After quantization and encoding: \n bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}"
            )

    return results_list, (h, w)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k, v in cur_ckt.items():
            if "encoder" in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)

        return model_list, quant_ckt


# Tensor quantization and de-quantization
def quant_tensor(t, bits=8):
    tmin_scale_list = []
    # quantize over the whole tensor, or along each dimenstion
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / (2**bits - 1)
    tmin_scale_list.append([t_min, scale])
    for axis in range(t.dim()):
        t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
        if t_min.nelement() / t.nelement() < 0.02:
            scale = (t_max - t_min) / (2**bits - 1)
            # tmin_scale_list.append([t_min, scale])
            tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)])
    # import pdb; pdb.set_trace; from IPython import embed; embed()

    quant_t_list, new_t_list, err_t_list = [], [], []
    for t_min, scale in tmin_scale_list:
        t_min, scale = t_min.expand_as(t), scale.expand_as(t)
        quant_t = ((t - t_min) / (scale)).round().clamp(0, 2**bits - 1)
        new_t = t_min + scale * quant_t
        err_t = (t - new_t).abs().mean()
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append(err_t)

    # choose the best quantization
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_new_t = new_t_list[best_quant_idx]
    best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
    best_tmin = tmin_scale_list[best_quant_idx][0]
    best_scale = tmin_scale_list[best_quant_idx][1]
    quant_t = {"quant": best_quant_t, "min": best_tmin, "scale": best_scale}

    return quant_t, best_new_t


def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t["quant"], quant_t["min"], quant_t["scale"]
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t
"""
