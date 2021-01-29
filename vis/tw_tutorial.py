import tensorwatch as tw
import time
import torchvision


def save_test_ipynb():
    """
    通过指定 test.log，tw.Watcher.make_notebook() 生成 ipynb
    """
    # streams will be stored in test.log file
    w = tw.Watcher(filename='vis/test.log')  # 也决定了 test.ipynb 目录

    # create a stream for logging
    s = w.create_stream(name='metric1')

    # generate Jupyter Notebook to view real-time streams
    # 将观测变量存储到 notebook
    w.make_notebook()

    for i in range(1000):
        # write x,y pair
        s.write((i, i**2))
        time.sleep(1)


def save_model_arch():
    alexnet_model = torchvision.models.alexnet()

    # 在 ipynb 就能直接显示图片了
    # 和教程显示的图片不一样
    img = tw.draw_model(alexnet_model, [1, 3, 224, 224])
    # print(type(img))  # <class 'tensorwatch.model_graph.hiddenlayer.pytorch_draw_model.DotWrapper'>
    img.save('vis/alexnet.jpg')


if __name__ == '__main__':
    save_model_arch()
