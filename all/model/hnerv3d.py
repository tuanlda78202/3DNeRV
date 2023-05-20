import sys
import numpy as np
import torch
import torch.nn as nn

t = torch.rand(1, 576, 57600)
print(sys.getsizeof(t.storage()))


class HNeRV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split("_")]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split("_")]

        # BUILD Encoder LAYERS
        if len(args.enc_strds):  # HNeRV
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split("_")]
            c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(
                args.enc_strds
            )
            c_out_list[-1] = enc_dim2

            self.encoder = ConvNeXt(
                stage_blocks=enc_blks,
                strds=args.enc_strds,
                dims=c_out_list,
                drop_path_rate=0,
            )

            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            ch_in = enc_dim2

        else:
            ch_in = 2 * int(args.embed.split("_")[-1])
            self.pe_embed = PositionEncoding(args.embed)
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split("_")]

        # BUILD Decoder LAYERS
        decoder_layers = []
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(
            dec_block=False,
            conv_type="conv",
            ngf=ch_in,
            new_ngf=out_f,
            ks=0,
            strd=1,
            bias=True,
            norm=args.norm,
            act=args.act,
        )
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(args.dec_strds):
            reduction = sqrt(strd) if args.reduce == -1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                cur_blk = NeRVBlock(
                    dec_block=True,
                    conv_type=args.conv_type[1],
                    ngf=ngf,
                    new_ngf=new_ngf,
                    ks=min(ks_dec1 + 2 * i, ks_dec2),
                    strd=1 if j else strd,
                    bias=True,
                    norm=args.norm,
                    act=args.act,
                )
                decoder_layers.append(cur_blk)
                ngf = new_ngf

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1)
        self.out_bias = args.out_bias

    def forward(self, input, input_embed=None, encode_only=False):
        if input_embed != None:
            img_embed = input_embed
        else:
            if "pe" in self.embed:
                input = self.pe_embed(input[:, None]).float()
            img_embed = self.encoder(input)

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        embed_list = [img_embed]
        dec_start = time.time()
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = (
            output.view(n, -1, self.fc_h, self.fc_w, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(n, -1, self.fc_h * h, self.fc_w * w)
        )
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output)
            embed_list.append(output)

        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return img_out, embed_list, dec_time


class HNeRVDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc_h, self.fc_w = [torch.tensor(x) for x in [model.fc_h, model.fc_w]]
        self.out_bias = model.out_bias
        self.decoder = model.decoder
        self.head_layer = model.head_layer

    def forward(self, img_embed):
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = (
            output.view(n, -1, self.fc_h, self.fc_w, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(n, -1, self.fc_h * h, self.fc_w * w)
        )
        for layer in self.decoder[1:]:
            output = layer(output)
        output = self.head_layer(output)

        return OutImg(output, self.out_bias)
