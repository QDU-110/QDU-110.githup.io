import torch
print('111')
# import onnx
import functools
from options.train_options import TrainOptions
from models import networks
from torchsummary import summary
opt = TrainOptions().parse()
encoder = networks.define_E(opt.output_nc, 8, opt.nef, netE='resnet_128', norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain,vaeLike=True).cuda()

Generator = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                  norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                  init_gain=opt.init_gain,
                                  where_add=opt.where_add, upsample=opt.upsample).cuda()

# Generator = networks.MACGenerator(opt.input_nc, opt.output_nc, nz=8).cuda()
Discriminator = networks.define_D(1, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds).cuda()
# summary(encoder,(3,128,128,128))
input = torch.randn((1,1,128,128,128)).cuda()
output, outputVar = encoder(input)
fake = Generator(input, output)
di = Discriminator(fake)
# torch.onnx.export(
#     Generator,(input,output),
#     'Generator.onnx',
#     export_params=True,
#     opset_version=8,
# )
# model_file = 'Generator.onnx'
# onnx_model = onnx.load(model_file)
# onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
print('111')


