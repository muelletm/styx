
import stupid

stupid.setDataPath("../data")
d = stupid.model.decoder_vgg()
print(d.summary())