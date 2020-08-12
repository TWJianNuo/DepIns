from __future__ import absolute_import, division, print_function

from options import MonodepthOptions
from layers import *
import datasets
import networks
import inplaceShapeLoss_cuda


cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "..", "splits")

STEREO_SCALE_FACTOR = 5.4
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       encoder_dict['height'], encoder_dict['width'],[0], 4, is_train=False)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    dirmapping = {'l':'image_02', 'r':'image_03'}
    localgeomDict = dict()

    print("-> Computing predictions with size {}x{}".format(encoder_dict['width'], encoder_dict['height']))

    totloss = 0

    with torch.no_grad():
        for count in range(len(filenames)):
            data = dataset.__getitem__(count)
            input_color = data[("color", 0, 0)].unsqueeze(0).cuda()

            output = depth_decoder(encoder(input_color))
            _, preddepth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
            preddepth = preddepth * STEREO_SCALE_FACTOR
            htheta = output[("disp", 0)][:, 0:1, :, :] * 2 * np.pi
            vtheta = output[("disp", 0)][:, 1:2, :, :] * 2 * np.pi

            seq, frame, dir = filenames[count].split(' ')
            depthgt = pil.open(os.path.join(opt.kitti_gt_path, seq, dirmapping[dir], frame + '.png'))
            depthgt = np.array(depthgt).astype(np.float32) / 256.0
            depthgt = torch.from_numpy(depthgt).unsqueeze(0).unsqueeze(0).cuda()

            _, _, ch, cw = depthgt.shape

            acckey = str(ch) + '_' + str(cw)
            if acckey not in localgeomDict:
                kittiw = cw
                kittih = ch
                intrinsicKitti = np.array([
                    [0.58 * kittiw, 0, 0.5 * kittiw],
                    [0, 1.92 * kittih, 0.5 * kittih],
                    [0, 0, 1]], dtype=np.float32)
                localthetadesp = LocalThetaDesp(height=kittih, width=kittiw, batch_size=1, intrinsic=intrinsicKitti).cuda()
                localgeomDict[acckey] = localthetadesp

            rgbi = F.interpolate(input_color, [ch, cw], mode='bilinear', align_corners=True)
            hthetai = F.interpolate(htheta, [ch, cw], mode='bilinear', align_corners=True)
            vthetai = F.interpolate(vtheta, [ch, cw], mode='bilinear', align_corners=True)
            preddepthi = F.interpolate(preddepth, [ch, cw], mode='bilinear', align_corners=True)

            # hthetai, vthetai = localgeomDict[acckey].get_theta(preddepthi)

            ratioh, ratiohl, ratiov, ratiovl = localgeomDict[acckey].get_ratio(htheta=hthetai, vtheta=vthetai)

            # ratiohl = torch.zeros_like(ratiohl)
            # ratiovl = torch.zeros_like(ratiovl)


            logdepthd = torch.log(depthgt)
            valindic = depthgt > 0
            lossrec = torch.zeros_like(logdepthd)
            countsrec = torch.zeros_like(logdepthd)
            rndseeds = torch.rand_like(logdepthd)
            inplaceShapeLoss_cuda.inplaceShapeLoss_forward(logdepthd, ratiohl, ratiovl, valindic.int(), lossrec, countsrec, rndseeds, 30, 30)

            totloss = totloss + torch.sum(lossrec[lossrec > 0]) / torch.sum(lossrec > 0)

            # cm = plt.get_cmap('bwr')
            # xx, yy = np.meshgrid(range(cw), range(ch), indexing='xy')
            # lossrecnp = lossrec[0, 0, :, :].cpu().numpy()
            # valmask = np.abs(lossrecnp) > 0
            # z = lossrecnp[valmask]
            #
            # selector_pos = z > 0
            # selector_neg = z < 0
            #
            # bar = 0.005
            #
            # if np.sum(selector_pos) > 1:
            #     pos_bar = bar
            #     z[selector_pos] = z[selector_pos] / pos_bar / 2
            #
            # if np.sum(selector_neg) > 1:
            #     neg_bar = -bar
            #     z[selector_neg] = -z[selector_neg] / neg_bar / 2
            #
            # znormed = z + 0.5
            # colorMap = cm(znormed)[:, 0:3]
            #
            # plt.figure(figsize=(12, 9), dpi=120, facecolor='w', edgecolor='k')
            # plt.imshow(tensor2rgb(rgbi, ind=0))
            # plt.scatter(xx[valmask], yy[valmask], c=colorMap, s=8)
            # plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_shapeErrType', str(count) + '.png'))
            # plt.close()

            # hthetad, vthetad = localgeomDict[acckey].get_theta(depthmap=preddepthi)
            # ratiohd, ratiohld, ratiovd, ratiovld = localgeomDict[acckey].get_ratio(htheta=hthetad, vtheta=vthetad)
            # logdepthd = torch.log(preddepthi)
            # valindic = preddepthi > 0
            # lossrec = torch.zeros_like(logdepthd)
            # inplaceShapeLoss_cuda.inplaceShapeLoss_integration(logdepthd, ratiohld, ratiovld, valindic.int(), lossrec, 1, 1)

    totloss = totloss / len(filenames)
    print(totloss)

if __name__ == "__main__":
    options = MonodepthOptions()
    args = options.parse()
    evaluate(args)

