import torch
from skimage.color import rgb2hsv
from torch import nn

loss_mse = nn.MSELoss()

def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def caploss(J):

    hsv = np_to_torch(rgb2hsv(torch_to_np(J).transpose(1, 2, 0)))
    cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]
    cap_loss = loss_mse(cap_prior, torch.zeros_like(cap_prior))

    return cap_loss



if __name__ =="__main__":

    x=torch.randn(1,3,256,256).cuda()
    bsloss = caploss(x)
    print(bsloss)

    import cv2

    # 图像路径（注意：在字符串中用双反斜杠或加 r 使其成为原始字符串）
    image_path = r"E:\programs\ZSID\datasets\reals\haze\GT_100.png"

    # 读取图像（默认以 BGR 格式）
    image_bgr = cv2.imread(image_path)

    # 检查是否成功读取图像
    if image_bgr is None:
        print("无法读取图像，请检查路径是否正确。")
    else:
        # 转换到 HSV 颜色空间
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 分离 H, S, V 分量
        H, S, V = cv2.split(image_hsv)

        # 输出 S 和 V 分量的信息
        print("Saturation (S) channel shape:", S.shape)
        print("Value (V) channel shape:", V.shape)

        # 如需显示 S 或 V 分量，可以使用如下命令（需要 GUI 支持）：
        # cv2.imshow("Saturation (S)", S)
        # cv2.imshow("Value (V)", V)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("S_channel.png", S)
        cv2.imwrite("V_channel.png", V)




