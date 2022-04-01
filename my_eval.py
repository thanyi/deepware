"""
本py文件是运用在项目运行中的模型测试部分
功能是对一整个文件夹中的图片进行模型检测，如果比判断为deepfake的图片的概率大于50%，则判断为deepfake视频
"""
from scan import *
from torchvision import transforms
from utils.utils import evaluate
import dataset.dataset_conf as config


def prepare_model():
    # 模型的配置文件  地址是同级目录下的json文件 同时json文件里面的pretrain_path需要修改
    with open("config.json") as f:
        cfg = json.loads(f.read())

    arch = cfg['arch']
    margin = cfg['margin']
    face_size = (cfg['size'], cfg['size'])

    print(f'margin: {margin}, size: {face_size}, arch: {arch}')

    # 模型调用
    model_list = []
    model = EffNet(arch).to("cuda:0")
    checkpoint = torch.load(cfg['pretain_model_path'], map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint)
    del checkpoint

    model_list.append(model)
    deepware = Ensemble(model_list).eval().to("cuda:0")

    return deepware

def modelTest():

    datasetname = config.malicious_root.split('/')[-1]
    model = prepare_model()

    r_acc, auc ,con_mat= evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "test")
    print(config.model_name+"模型在{}数据集上的acc为：".format(datasetname) + str(r_acc))
    print(config.model_name+"模型在{}数据集上的auc为：".format(datasetname) + str(auc))
    print(config.model_name+"模型在{}数据集上的con_mat为：".format(datasetname) + str(con_mat))


if __name__ == '__main__':
    modelTest()