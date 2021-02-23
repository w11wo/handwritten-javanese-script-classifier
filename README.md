# Handwritten Javanese Script Classification
> A Convolutional Neural Network trained to predict handwritten Javanese Script.

## Background
### Aksara Jawa
*Aksara Jawa*, or the [Javanese Script](https://en.wikipedia.org/wiki/Javanese_script) is the core of writing of the Javanese language and has influenced various other regional languages such as Sundanese, Madurese, etc. The script is now rarely used on a daily basis, but is sometimes taught in local schools in certain provinces of Indonesia.

### Specific Form of Aksara
The Javanese Script which we will be classifiying is specifically [Aksara Wyanjana](https://en.wikipedia.org/wiki/Javanese_script#Wyanjana)'s *Nglegena*, or its basic characters. The list consists of 20 basic characters, without their respective *Pasangan* characters.

### Dataset
Since I have not been able to find a handwritten Javanese Script dataset on the internet, I have decided to contact one of my English highschool teachers who has once showed my class her ability to write Javanese Script. The characters were written on paper, scanned, and edited manually. Credits to **Mm. Martha Indrati** for the help!

### Image Classification
This project is very much inspired from datasets like [MNIST](http://yann.lecun.com/exdb/mnist/) and [QMNIST](https://github.com/facebookresearch/qmnist) which are handwritten digits and is a go-to dataset for starting to learn image classification. The end goal of this project is to be able to create a deep learning model which will be able to classify handwritten Javanese Script to a certain degree of accuracy.

## Code
The main framework to be used is fastai-v2, which sits on top of PyTorch. Fastai-v2 is still under development as of the time of this writing, but is ready to be used for basic image classification tasks.


```python
from fastai2.vision.all import *
import torch
```

### Load Data
The data has been grouped per class folder, which we'll load up and later split into training (70%) and validation (30%) images.


```python
path = Path("handwritten-javanese-script-dataset")
```

Notice we're using a small batch size of 5, mainly because we only have 200 images in total.

Here we'll apply cropping and resizing as transformations to our image since most of the characters do not fully occupy the image size. Additionally, we'll resize to 128px.


```python
dblock = DataBlock(blocks     = (ImageBlock(cls=PILImageBW), CategoryBlock),
                   get_items  = get_image_files,
                   splitter   = GrandparentSplitter(valid_name='val'),
                   get_y      = parent_label,
                   item_tfms  = [CropPad(90), Resize(128, method=ResizeMethod.Crop)])
```


```python
dls = dblock.dataloaders(path, bs=5, num_workers=0)
```


```python
dls.show_batch()
```


    
![png](images/output_9_0.png)
    


There are only 20 types of characters in the type of Aksara which we'll be classifying.


```python
dls.vocab
```




    (#20) ['ba','ca','da','dha','ga','ha','ja','ka','la','ma'...]



### Model
We'll be using **XResNet50** as the model, which is based on the [Bag of Tricks paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) and is an "extension" to the [ResNet50](https://arxiv.org/abs/1512.03385) architecture. We'll pass our data, tell which metrics we'd like to observe, utilize `LabelSmoothingCrossEntropy`, and add `MixUp` as our callback.


```python
learn = Learner(dls, xresnet50(c_in=1, n_out=dls.c), metrics=accuracy, loss_func=LabelSmoothingCrossEntropy(), cbs=MixUp)
```

### Training Model
With all things in place, let's finally train the model to learn from the given dataset and predict which class the image belongs to.


```python
learn.lr_find()
```








    SuggestedLRs(lr_min=0.0003019951749593019, lr_steep=6.309573450380412e-07)




    
![png](images/output_15_2.png)
    



```python
learn.fit_one_cycle(30, 3e-4, cbs=SaveModelCallback(monitor='accuracy', fname='best_model'), wd=0.4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.067268</td>
      <td>3.108827</td>
      <td>0.050000</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.929908</td>
      <td>2.669373</td>
      <td>0.333333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.769148</td>
      <td>2.293764</td>
      <td>0.383333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.588481</td>
      <td>2.215439</td>
      <td>0.316667</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.416248</td>
      <td>2.324036</td>
      <td>0.283333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.324458</td>
      <td>1.983255</td>
      <td>0.533333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.189000</td>
      <td>2.105889</td>
      <td>0.383333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.078479</td>
      <td>2.350886</td>
      <td>0.333333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.922369</td>
      <td>2.823610</td>
      <td>0.216667</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.790820</td>
      <td>1.584189</td>
      <td>0.650000</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1.683853</td>
      <td>1.509675</td>
      <td>0.583333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>11</td>
      <td>1.598790</td>
      <td>1.570487</td>
      <td>0.650000</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>12</td>
      <td>1.528586</td>
      <td>1.256149</td>
      <td>0.833333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>13</td>
      <td>1.484508</td>
      <td>1.623523</td>
      <td>0.566667</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>14</td>
      <td>1.437240</td>
      <td>1.340925</td>
      <td>0.750000</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>15</td>
      <td>1.345987</td>
      <td>1.138785</td>
      <td>0.816667</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>16</td>
      <td>1.350891</td>
      <td>1.370259</td>
      <td>0.716667</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1.297572</td>
      <td>1.453033</td>
      <td>0.666667</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>18</td>
      <td>1.318248</td>
      <td>1.330522</td>
      <td>0.750000</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1.263931</td>
      <td>1.023822</td>
      <td>0.900000</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>20</td>
      <td>1.247242</td>
      <td>1.063768</td>
      <td>0.900000</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>21</td>
      <td>1.234829</td>
      <td>1.009032</td>
      <td>0.933333</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>22</td>
      <td>1.203268</td>
      <td>0.968369</td>
      <td>0.950000</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>23</td>
      <td>1.178766</td>
      <td>0.965601</td>
      <td>0.916667</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>24</td>
      <td>1.156069</td>
      <td>0.939599</td>
      <td>0.933333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>25</td>
      <td>1.183693</td>
      <td>0.943586</td>
      <td>0.933333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>26</td>
      <td>1.166053</td>
      <td>0.933629</td>
      <td>0.933333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>27</td>
      <td>1.162939</td>
      <td>0.936014</td>
      <td>0.933333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>28</td>
      <td>1.132883</td>
      <td>0.936722</td>
      <td>0.933333</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>29</td>
      <td>1.138776</td>
      <td>0.946842</td>
      <td>0.933333</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>


    Better model found at epoch 0 with accuracy value: 0.05000000074505806.
    Better model found at epoch 1 with accuracy value: 0.3333333432674408.
    Better model found at epoch 2 with accuracy value: 0.38333332538604736.
    Better model found at epoch 5 with accuracy value: 0.5333333611488342.
    Better model found at epoch 9 with accuracy value: 0.6499999761581421.
    Better model found at epoch 12 with accuracy value: 0.8333333134651184.
    Better model found at epoch 19 with accuracy value: 0.8999999761581421.
    Better model found at epoch 21 with accuracy value: 0.9333333373069763.
    Better model found at epoch 22 with accuracy value: 0.949999988079071.
    


```python
learn.recorder.plot_loss()
```


    
![png](images/output_17_0.png)
    



```python
learn.save('stage-1')
```

### Analyze Results
After training, let's see how well our model learned. Any incorrect prediction in a random batch will have its label colored red.


```python
learn.show_results()
```






    
![png](images/output_20_1.png)
    


Instead of only viewing a batch, let's analyze the results from the entire validation dataset.


```python
interp =  ClassificationInterpretation.from_learner(learn)
```





This confusion matrix lists all the actual versus predicted labels. The darker the blue on the diagonal line, the better our model is at predicting.


```python
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
```


    
![png](images/output_24_0.png)
    


On the other hand, this type of interpretation shows several of the predicted images, what our model thinks it is, and how confident it is with that prediction.


```python
interp.plot_top_losses(9, figsize=(10,9))
```


    
![png](images/output_26_0.png)
    


### Predicting External Images
To see how our model's regularization fairs, let's attempt to feed it an external data and see what it predicted.


```python
from PIL import Image
```


```python
def open_image_bw_resize(source) -> PILImageBW:
    return PILImageBW(Image.open(source).resize((128,128)).convert('L'))
```

The following character is supposed to be **ma** and was picked randomly from available images on the internet.


```python
test0 = open_image_bw_resize('test-images/test-image-0.jpg')
test0.show()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e8960ffaf0>




    
![png](images/output_31_1.png)
    


Feed it through the model and see its output.


```python
learn.predict(test0)[0]
```








    'ma'



Luckily, the model was able to predict the character correctly. To challenge the model even more, I tried to write Javanese Script characters myself and see what the model predicts. Do note that I do not have any background in writing Javanese Scripts, so pardon my skills.

The following character is supposed to be **ca**.


```python
test1 = open_image_bw_resize('test-images/test-image-1.jpg')
test1.show()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e895ef6610>




    
![png](images/output_35_1.png)
    



```python
learn.predict(test1)[0]
```








    'ca'



This character is supposed to be **wa**.


```python
test2 = open_image_bw_resize('test-images/test-image-2.jpg')
test2.show()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e8c2a21580>




    
![png](images/output_38_1.png)
    



```python
learn.predict(test2)[0]
```








    'ca'



Well that's an incorrect guess, which is reasonable firstly because of my poor handwriting skills, and secondly the model was trained on a person's particular style of handwriting - which in this case is my teacher's. There could be many other factors which caused the incorrect guess, such as overfitting by the model, small dataset and possibly more.

## Closing Remarks
There are several possible improvements which could be made, one of which is to increase the variety and the size of the dataset, since the model is only training on a single person's handwriting. It'll be better in terms of regularization to add other people's handwriting into the mix as well.

That's it for this mini project of mine. Thanks for your time and I hope you've learned something!
