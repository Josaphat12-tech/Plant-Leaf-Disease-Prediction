# Plant-Leaf-Disease-Prediction

Abstract
Plants are a major source of food for the world population. Plant diseases contribute to production loss, which can be tackled with continuous monitoring. Manual plant disease monitoring is both laborious and error-prone. Early detection of plant diseases using computer vision and artificial intelligence (AI) can help to reduce the adverse effects of diseases and also helps to overcome the shortcomings of continuous human monitoring. In this study, we have extensively studied the performance of the different state-of-the-art convolutional neural networks (CNNs) classification network architectures i.e. ResNet18, MobileNet, DenseNet201, and InceptionV3 on 18,162 plain tomato leaf images to classify tomato diseases. The comparative performance of the models for the binary classification (healthy and unhealthy leaves), six-class classification (healthy and various groups of diseased leaves), and ten-class classification (healthy and various types of unhealthy leaves) are also reported. InceptionV3 showed superior performance for the binary classification using plain leaf images with an accuracy of 99.2%. DenseNet201 also outperform for six-class classification with an accuracy of 97.99%. Finally, DenseNet201 achieved an accuracy of 98.05% for ten-class classification. It can be concluded that deep architectures performed better at classifying the diseases for the three experiments. The performance of each of the experimental studies reported in this work outperforms the existing literature.

Keywords
Smart agricultureautomatic plant disease detectiondeep learningCNNclassification
Author Information
Show +
1. Introduction
Thousands of years ago, the development of agriculture led to the domestication of main food crops and animals today. One of the major global problems that humanity faces today is food insecurity [1] of which plant diseases are a major cause [2]. According to one estimate, plant diseases collectively account for a crop yield loss of around 16% globally [3]. The global potential loss from pests is estimated to be around 50% for wheat and 26–29% for soybean [3]. Plant pathogens are classified into major groups of fungi, fungus-like organisms, bacteria, virus, viroid, virus-like organism, nematodes, protozoa, algae, and parasitic plants. Artificial intelligence (AI), machine learning (ML), and computer vision have provided significant help in numerous applications including power prediction from renewable resources [4, 5] and biomedical applications [6, 7]. The application of AI has seen a great boost during the COVID-19 pandemic period for the detection of lung-related diseases [8, 9, 10, 11] and other prognostic applications [12]. Similar advanced technology can be used to mitigate the adverse effects of plant diseases by their early-stage detection and diagnosis. Recently, the application of AI and computer vision to automatic detection and diagnosis of plant diseases is being extensively studied because manual plant disease monitoring is tedious, time-consuming, and labor-intensive. Sidharth et al. [13] applied a Bacterial Foraging Optimization-based Radial Basis Function Network (BRBFNN) to automatically identify and classify plant disease achieving the classification accuracy of 83.07%. Convolutional neural network (CNN) is a very popular neural network architecture that is used successfully for a variety of computer vision tasks in diverse fields [14]. CNN architecture and its different variants have been utilized by researchers for the classification and detection of plant diseases. Sunayana et al. [15] compared different CNN architectures for disease detection in potato and mango leaves achieving an accuracy of 98.33% for AlexNet and 90.85% for a shallow CNN model. Guan et al. [15, 16] used a pre-trained VGG16 model to estimate the disease severity in apple plants and achieved an accuracy of 90.40%. Jihen et al. [17] used LeNet [18] model to classify healthy and diseased banana leaves and achieved an accuracy of 99.72%.

Tomato is a major food crop across the globe with a per capita consumption of 20 kilograms per year and represents about 15% of average total vegetable consumption. North America is consuming 42 kilograms of tomatoes per capita per year while Europe is consuming 31 kilograms of tomatoes per capita per year [19, 20]. To meet the global demand for tomatoes, it is imperative to devise techniques for improving crop yield and early detection of pests, bacterial, and viral infections. Several works have been done in employing artificial intelligence-based techniques to improve tomato plants’ survival by early detection of diseases and subsequent disease management. Manpreet et al. [21] used a pre-trained CNN-based architecture known as Residual Network or commonly called ResNet to classify seven tomato diseases with an accuracy of 98.8%. Rahman et al. [22] proposed a deep learning-based fully-connected network to classify Bacterial Spot, Late Blight, and Septorial Spot disease from tomato leaf images and achieved an accuracy of 99.25%. Fuentes et al. [23] to classify ten diseases from tomato leaves images considered three main families of detectors: Faster Region-based Convolutional Neural Network (Faster R-CNN), Region-based Fully Convolutional Network (R-FCN), and Single Shot Multibox Detector (SSD). These detectors were combined with different variants of deep feature extractors VGG16, ResNet50, and ResNet152 for Faster R-CNN, ResNet-50 for SSD, and ResNet-50 for R-FCN for real-time disease and pests’ recognition and achieved the highest Average Precision of 83% with VGG16 on top of FRCNN. Agarwal et al. [24] proposed a Tomato Leaf Disease Detection (ToLeD) model, a CNN-based architecture for the classification of ten diseases from tomato leaves images achieving an accuracy of 91.2%. Durmuş et al. [25] evaluated AlexNet and SqueezeNet architectures for the classification of ten diseases from tomato leaves images and achieved an accuracy of 95.5%. Although the disease classification and detection in plant leaves are well-studied in tomatoes and other plants, the reliability of leaf images with varying back-ground on large image classes are not well-studied, since the real-world images can vary greatly in terms of lighting conditions, image quality, orientation, etc. [18].

This chapter has the following main contributions: (i) Investigation of the classification tasks for different variants of CNN architecture for binary and different multi-class classifications of tomato diseases. Several experiments employing different CNN architectures were conducted on raw images. Three different types of classifications were done in this work: (a) Binary classification of healthy and diseased leaves, (b) Six-class classification of healthy and four diseased leaves, and finally, (c) Ten-class classification with healthy and 9 different diseases classes. (ii) The performance achieved in this work outperforms the existing state-of-the-art works in this domain.

The rest of the chapter is organized in the following manner: Section 1 gives a brief introduction, literature review, and motivation for the study. Section 2 describes the different types of plant pathogens. Section 3 provides the methodology of the study with technical details such as the dataset description, pre-processing techniques, and details of the experiments. Section 4 reports the results of the studies followed by discussions in Section 5 and finally, the conclusion is provided in Section 6.

Advertisement
2. Pathogens of tomato leaves
Fungi is the predominant plant pathogens and it can cause multiple diseases including early blight, septoria leaf spot, target spot, and leaf mold. Fungi can attack plants through different sources such as infected soil and seeds. Fungal infections can spread from one plant to another through animals, humans, machinery, and soil contamination. The fungal attack vectors include plant pruning wounds, insects, leaf stomata, and others. The early blight disease of tomato plants is caused by the fungus, which affects the plant leaves. If it affects the seedlings’ basal stems, adult plant’s stem, and fruits, it is called collar rot, stem lesion, and fruit rot, respectively [26, 27]. Numerous methods have been devised for the control of early blight but the most effective methods are cultural control i.e. efficient soil, nutrients, and crop management to reduce infections and also with the use of fungicidal chemicals. Septoria leaf spot of tomato plants is caused by fungus [28, 29], which releases tomatinase enzyme that speeds up the degradation of tomato steroidal glycoalkaloids α-tomatine [30, 31]. The target spot disease of tomato plants is caused by the fungus [32, 33]. Symptoms of target spot disease in tomato plants are necrotic lesions of light brown color in the center [34, 35]. The lesions spread to a larger blighted leaf area and result in early defoliation [34, 35]. The target spot also damages the fruit directly by entering into the fruit pulp [34, 35]. The leaf mold disease of plants is caused by the fungus [36, 37]. It occurs during periods of extended leaf wetness. Bacteria is also a major plant pathogen. Bacteria enter plants through wounds such as insect bites, pruning, cuts, and also through natural openings such as stomata. Plant’s surrounding environmental conditions such as temperature, humidity, soil conditions, availability of nutrients, weather conditions, and airflow are important factors in determining the bacterial growth on the plant and the consequent damage. Bacterial spot is a plant disease caused by bacteria [38, 39]. Molds are also a major cause of plant diseases. Late blight disease of tomato and potato plants is caused by mold [40, 41]. The appearance of dark uneven blemishes on leaves tips and plant stems are a few of the symptoms. Tomato yellow leaf curl virus (TYLCV) is a devastating virus causing tomato disease. This virus attacks the plant through another insect. Although tomato plants are unhealthy diseased leaves and iii) ten-class classification of healthy and various diseased leaves. In study II, different types of tomato leaf diseases are classified into the group of diseases while in study III, different classes of unhealthy and healthy leaf images were classified. Similar experiments the primary hosts for the virus, this viral infection has been reported in several other plants including beans and pepper, tobacco, potatoes, and eggplants [42, 43]. In the last few decades, due to the rapid spread of the disease, the research focus has been shifted to damage control of yellow leaf curl disease [44, 45, 46, 47]. Another viral disease that specifically affects tomato plants is caused by Tomato mosaic virus (ToMV). This virus is found worldwide and affects not only tomatoes but other plants as well. Symptoms of ToMV infection include twisting and fern-like appearance of leaves, damaged fruit with yellow patches, and necrotic blemishes [48, 49].

Advertisement
3. Methodology
The overall methodology of the study of the paper is summarized in Figure 1. This study used tomato leaf data from the plant village dataset [50, 51], where tomato leaf images are provided. As explained earlier, the paper has three different studies: (i) binary classification of healthy and unhealthy leaves; (ii) six-class classification of healthy and different disease group leaves were conducted; and (iii) ten class of healthy and several different diseased leaves were carried out. The classification is done using pre-trained networks- ResNet18, MobilenetV2, InceptionV3, and DenseNet201 that have been comparatively successful in previous publications [8, 10, 11, 52, 53, 54, 55, 56, 57].

![image](https://user-images.githubusercontent.com/85907186/203120576-ba039b1c-1af6-4c97-9676-2abc7166ec50.png)

Figure 1.
Overall Methodology of the study.
3.1 Datasets description
In this study, plant village tomato leaf images dataset was used [50, 51], where 18,162 tomato leaf images are available. All images were divided into 10 different classes, where one class is healthy and the other nine classes are unhealthy (such as- bacterial spot, early blight, leaf mold, septoria leaf spot, target spot, two-spotted spider mite, late bright mold, mosaic virus, and yellow leaf curl virus), and 9 unhealthy classes are categorized into five subgroups (namely-bacterial, viral, fungal, mold and mite disease). Some sample tomato leaf images, for healthy and different unhealthy classes from plant village dataset are shown in Figure 2. Moreover, a detailed description of the number of images in the dataset is also shown in Table 1, which is useful for classification tasks discussed in detail in the next section.
![image](https://user-images.githubusercontent.com/85907186/203120717-584cccbd-3c00-47cf-a362-b66d719d5db0.png)
3.2 Preprocessing
3.2.1 Resizing and normalizing
The various CNN network has input image size requirements. Thus, the images were resized to 299 × 299 for Inceptionv3 and 224 × 224 for Resnet18, MobilenetV2, and DenseNet201. Using the mean and standard deviation of the images of the dataset, z-score normalization was used to normalize the images.

3.2.2 Augmentation
Training with an imbalanced dataset will result in a biased model because the dataset is not balanced and does not contain a comparable number of images for the various categories. As a result, data augmentation can aid in the creation of a similar number of images in each class, resulting in reliable results, as reported in numerous recent publications [6, 7, 8, 9, 10, 11]. To align the training images, three augmentation techniques (rotation, scaling, and translation) were used. The images were rotated in a clockwise and counterclockwise direction with an angle of 5 to 15 degrees for image augmentation. The scaling process involves enlarging or shrinking the image’s frame size, and 2.5 percent to 10% image magnifications were used in this analysis. Image translation was accomplished by converting images by 5–20% horizontally and vertically.

3.3 Experiments
Four pre-trained CNN models were investigated that were originally trained on ImageNet Database [58] to classify tomato leaf images. Three different classification experiments were carried out in this study. Tables 2–4 summarize the details of the images in the experiments for three different classification of leaf images separately. Two of the four pre-trained networks are shallow (MobilenetV2, and ResNet18), while the other two are deep (Inceptionv3, and DenseNet201) to see whether shallow and deep networks are appropriate for this application. Table 5 presents a summary of the parameters (Batch size (BS), Learning rate (LR), Epochs (E), Epochs patience (EP), Loss function (LF), Optimizer (OP)) for classification in experiments.

Abstract
Plants are a major source of food for the world population. Plant diseases contribute to production loss, which can be tackled with continuous monitoring. Manual plant disease monitoring is both laborious and error-prone. Early detection of plant diseases using computer vision and artificial intelligence (AI) can help to reduce the adverse effects of diseases and also helps to overcome the shortcomings of continuous human monitoring. In this study, we have extensively studied the performance of the different state-of-the-art convolutional neural networks (CNNs) classification network architectures i.e. ResNet18, MobileNet, DenseNet201, and InceptionV3 on 18,162 plain tomato leaf images to classify tomato diseases. The comparative performance of the models for the binary classification (healthy and unhealthy leaves), six-class classification (healthy and various groups of diseased leaves), and ten-class classification (healthy and various types of unhealthy leaves) are also reported. InceptionV3 showed superior performance for the binary classification using plain leaf images with an accuracy of 99.2%. DenseNet201 also outperform for six-class classification with an accuracy of 97.99%. Finally, DenseNet201 achieved an accuracy of 98.05% for ten-class classification. It can be concluded that deep architectures performed better at classifying the diseases for the three experiments. The performance of each of the experimental studies reported in this work outperforms the existing literature.

Keywords
Smart agricultureautomatic plant disease detectiondeep learningCNNclassification
Author Information
Show +
1. Introduction
Thousands of years ago, the development of agriculture led to the domestication of main food crops and animals today. One of the major global problems that humanity faces today is food insecurity [1] of which plant diseases are a major cause [2]. According to one estimate, plant diseases collectively account for a crop yield loss of around 16% globally [3]. The global potential loss from pests is estimated to be around 50% for wheat and 26–29% for soybean [3]. Plant pathogens are classified into major groups of fungi, fungus-like organisms, bacteria, virus, viroid, virus-like organism, nematodes, protozoa, algae, and parasitic plants. Artificial intelligence (AI), machine learning (ML), and computer vision have provided significant help in numerous applications including power prediction from renewable resources [4, 5] and biomedical applications [6, 7]. The application of AI has seen a great boost during the COVID-19 pandemic period for the detection of lung-related diseases [8, 9, 10, 11] and other prognostic applications [12]. Similar advanced technology can be used to mitigate the adverse effects of plant diseases by their early-stage detection and diagnosis. Recently, the application of AI and computer vision to automatic detection and diagnosis of plant diseases is being extensively studied because manual plant disease monitoring is tedious, time-consuming, and labor-intensive. Sidharth et al. [13] applied a Bacterial Foraging Optimization-based Radial Basis Function Network (BRBFNN) to automatically identify and classify plant disease achieving the classification accuracy of 83.07%. Convolutional neural network (CNN) is a very popular neural network architecture that is used successfully for a variety of computer vision tasks in diverse fields [14]. CNN architecture and its different variants have been utilized by researchers for the classification and detection of plant diseases. Sunayana et al. [15] compared different CNN architectures for disease detection in potato and mango leaves achieving an accuracy of 98.33% for AlexNet and 90.85% for a shallow CNN model. Guan et al. [15, 16] used a pre-trained VGG16 model to estimate the disease severity in apple plants and achieved an accuracy of 90.40%. Jihen et al. [17] used LeNet [18] model to classify healthy and diseased banana leaves and achieved an accuracy of 99.72%.

Tomato is a major food crop across the globe with a per capita consumption of 20 kilograms per year and represents about 15% of average total vegetable consumption. North America is consuming 42 kilograms of tomatoes per capita per year while Europe is consuming 31 kilograms of tomatoes per capita per year [19, 20]. To meet the global demand for tomatoes, it is imperative to devise techniques for improving crop yield and early detection of pests, bacterial, and viral infections. Several works have been done in employing artificial intelligence-based techniques to improve tomato plants’ survival by early detection of diseases and subsequent disease management. Manpreet et al. [21] used a pre-trained CNN-based architecture known as Residual Network or commonly called ResNet to classify seven tomato diseases with an accuracy of 98.8%. Rahman et al. [22] proposed a deep learning-based fully-connected network to classify Bacterial Spot, Late Blight, and Septorial Spot disease from tomato leaf images and achieved an accuracy of 99.25%. Fuentes et al. [23] to classify ten diseases from tomato leaves images considered three main families of detectors: Faster Region-based Convolutional Neural Network (Faster R-CNN), Region-based Fully Convolutional Network (R-FCN), and Single Shot Multibox Detector (SSD). These detectors were combined with different variants of deep feature extractors VGG16, ResNet50, and ResNet152 for Faster R-CNN, ResNet-50 for SSD, and ResNet-50 for R-FCN for real-time disease and pests’ recognition and achieved the highest Average Precision of 83% with VGG16 on top of FRCNN. Agarwal et al. [24] proposed a Tomato Leaf Disease Detection (ToLeD) model, a CNN-based architecture for the classification of ten diseases from tomato leaves images achieving an accuracy of 91.2%. Durmuş et al. [25] evaluated AlexNet and SqueezeNet architectures for the classification of ten diseases from tomato leaves images and achieved an accuracy of 95.5%. Although the disease classification and detection in plant leaves are well-studied in tomatoes and other plants, the reliability of leaf images with varying back-ground on large image classes are not well-studied, since the real-world images can vary greatly in terms of lighting conditions, image quality, orientation, etc. [18].

This chapter has the following main contributions: (i) Investigation of the classification tasks for different variants of CNN architecture for binary and different multi-class classifications of tomato diseases. Several experiments employing different CNN architectures were conducted on raw images. Three different types of classifications were done in this work: (a) Binary classification of healthy and diseased leaves, (b) Six-class classification of healthy and four diseased leaves, and finally, (c) Ten-class classification with healthy and 9 different diseases classes. (ii) The performance achieved in this work outperforms the existing state-of-the-art works in this domain.

The rest of the chapter is organized in the following manner: Section 1 gives a brief introduction, literature review, and motivation for the study. Section 2 describes the different types of plant pathogens. Section 3 provides the methodology of the study with technical details such as the dataset description, pre-processing techniques, and details of the experiments. Section 4 reports the results of the studies followed by discussions in Section 5 and finally, the conclusion is provided in Section 6.

Advertisement
2. Pathogens of tomato leaves
Fungi is the predominant plant pathogens and it can cause multiple diseases including early blight, septoria leaf spot, target spot, and leaf mold. Fungi can attack plants through different sources such as infected soil and seeds. Fungal infections can spread from one plant to another through animals, humans, machinery, and soil contamination. The fungal attack vectors include plant pruning wounds, insects, leaf stomata, and others. The early blight disease of tomato plants is caused by the fungus, which affects the plant leaves. If it affects the seedlings’ basal stems, adult plant’s stem, and fruits, it is called collar rot, stem lesion, and fruit rot, respectively [26, 27]. Numerous methods have been devised for the control of early blight but the most effective methods are cultural control i.e. efficient soil, nutrients, and crop management to reduce infections and also with the use of fungicidal chemicals. Septoria leaf spot of tomato plants is caused by fungus [28, 29], which releases tomatinase enzyme that speeds up the degradation of tomato steroidal glycoalkaloids α-tomatine [30, 31]. The target spot disease of tomato plants is caused by the fungus [32, 33]. Symptoms of target spot disease in tomato plants are necrotic lesions of light brown color in the center [34, 35]. The lesions spread to a larger blighted leaf area and result in early defoliation [34, 35]. The target spot also damages the fruit directly by entering into the fruit pulp [34, 35]. The leaf mold disease of plants is caused by the fungus [36, 37]. It occurs during periods of extended leaf wetness. Bacteria is also a major plant pathogen. Bacteria enter plants through wounds such as insect bites, pruning, cuts, and also through natural openings such as stomata. Plant’s surrounding environmental conditions such as temperature, humidity, soil conditions, availability of nutrients, weather conditions, and airflow are important factors in determining the bacterial growth on the plant and the consequent damage. Bacterial spot is a plant disease caused by bacteria [38, 39]. Molds are also a major cause of plant diseases. Late blight disease of tomato and potato plants is caused by mold [40, 41]. The appearance of dark uneven blemishes on leaves tips and plant stems are a few of the symptoms. Tomato yellow leaf curl virus (TYLCV) is a devastating virus causing tomato disease. This virus attacks the plant through another insect. Although tomato plants are unhealthy diseased leaves and iii) ten-class classification of healthy and various diseased leaves. In study II, different types of tomato leaf diseases are classified into the group of diseases while in study III, different classes of unhealthy and healthy leaf images were classified. Similar experiments the primary hosts for the virus, this viral infection has been reported in several other plants including beans and pepper, tobacco, potatoes, and eggplants [42, 43]. In the last few decades, due to the rapid spread of the disease, the research focus has been shifted to damage control of yellow leaf curl disease [44, 45, 46, 47]. Another viral disease that specifically affects tomato plants is caused by Tomato mosaic virus (ToMV). This virus is found worldwide and affects not only tomatoes but other plants as well. Symptoms of ToMV infection include twisting and fern-like appearance of leaves, damaged fruit with yellow patches, and necrotic blemishes [48, 49].

Advertisement
3. Methodology
The overall methodology of the study of the paper is summarized in Figure 1. This study used tomato leaf data from the plant village dataset [50, 51], where tomato leaf images are provided. As explained earlier, the paper has three different studies: (i) binary classification of healthy and unhealthy leaves; (ii) six-class classification of healthy and different disease group leaves were conducted; and (iii) ten class of healthy and several different diseased leaves were carried out. The classification is done using pre-trained networks- ResNet18, MobilenetV2, InceptionV3, and DenseNet201 that have been comparatively successful in previous publications [8, 10, 11, 52, 53, 54, 55, 56, 57].


Figure 1.
Overall Methodology of the study.
3.1 Datasets description
In this study, plant village tomato leaf images dataset was used [50, 51], where 18,162 tomato leaf images are available. All images were divided into 10 different classes, where one class is healthy and the other nine classes are unhealthy (such as- bacterial spot, early blight, leaf mold, septoria leaf spot, target spot, two-spotted spider mite, late bright mold, mosaic virus, and yellow leaf curl virus), and 9 unhealthy classes are categorized into five subgroups (namely-bacterial, viral, fungal, mold and mite disease). Some sample tomato leaf images, for healthy and different unhealthy classes from plant village dataset are shown in Figure 2. Moreover, a detailed description of the number of images in the dataset is also shown in Table 1, which is useful for classification tasks discussed in detail in the next section.


Figure 2.
Sample images of healthy and different unhealthy tomato leaves from the plant village database [3].
Class	Unhealthy	Healthy
Fungi	Bacteria	Mold	Virus	Mite
Sub Class	Early blight (1000)	Bacterial spot (2127)	Late bright mold (1910)	Tomato Yellow Leaf Curl Virus (5357)	Two spotted spider mite (1676)	Healthy (1591)
Septoria leaf spot (1771)
Tomato Mosaic Virus (373)
Target spot (1404)
Leaf mold(952)
Total Tomato Leaf Images (18,162)
Table 1.
The number of tomato leaf images for healthy and different unhealthy classes.

3.2 Preprocessing
3.2.1 Resizing and normalizing
The various CNN network has input image size requirements. Thus, the images were resized to 299 × 299 for Inceptionv3 and 224 × 224 for Resnet18, MobilenetV2, and DenseNet201. Using the mean and standard deviation of the images of the dataset, z-score normalization was used to normalize the images.

3.2.2 Augmentation
Training with an imbalanced dataset will result in a biased model because the dataset is not balanced and does not contain a comparable number of images for the various categories. As a result, data augmentation can aid in the creation of a similar number of images in each class, resulting in reliable results, as reported in numerous recent publications [6, 7, 8, 9, 10, 11]. To align the training images, three augmentation techniques (rotation, scaling, and translation) were used. The images were rotated in a clockwise and counterclockwise direction with an angle of 5 to 15 degrees for image augmentation. The scaling process involves enlarging or shrinking the image’s frame size, and 2.5 percent to 10% image magnifications were used in this analysis. Image translation was accomplished by converting images by 5–20% horizontally and vertically.

3.3 Experiments
Four pre-trained CNN models were investigated that were originally trained on ImageNet Database [58] to classify tomato leaf images. Three different classification experiments were carried out in this study. Tables 2–4 summarize the details of the images in the experiments for three different classification of leaf images separately. Two of the four pre-trained networks are shallow (MobilenetV2, and ResNet18), while the other two are deep (Inceptionv3, and DenseNet201) to see whether shallow and deep networks are appropriate for this application. Table 5 presents a summary of the parameters (Batch size (BS), Learning rate (LR), Epochs (E), Epochs patience (EP), Loss function (LF), Optimizer (OP)) for classification in experiments.

Database	Types	Total No. of images/ class	For Both Segmented and Unsegmented experiment
Train set count/fold	Validation set count/fold	Test set count/ fold
Plant village dataset	Healthy	1591	1147*10 = 11470	127	317
Unhealthy (9 diseases)	16570	11930	1326	3314
Table 2.
Summary of the binary classification experiment.

Database	Types	Count of images/class	For Both Segmented and Unsegmented experiment
Train set count/fold	Validation set count/fold	Test set count/fold
Plant village dataset	Healthy	1591	1147*3 = 3441	127	317
Fungi	5127	3692	410	1025
Bacteria	2127	1532*2 = 3064	170	425
Mold	1910	1375*3 = 4125	153	382
Virus	5730	4126	458	1146
Mite	1676	1207*3 = 3621	134	335
Table 3.
Summary of the six-class classification experiment.

Database	Types	Count of images/class	For Both Segmented and Unsegmented experiment
Train set count/fold	Validation set count/fold	Test set count/fold
Plant village dataset	Healthy	1591	1147*3 = 3441	127	317
Early blight	1000	720*5 = 3600	80	200
Septoria leaf spot	1771	1275*3 = 3825	142	354
Target spot	1404	1011*3 = 3033	112	281
Leaf mold	952	686*5 = 3430	76	190
Bacterial spot	2127	1532*2 = 3064	170	425
Late bright mold	1910	1375*3 = 4125	153	382
Tomato Yellow Leaf Curl Virus	5357	3857	429	1071
Tomato Mosaic Virus	373	268*13 = 3484	30	75
Table 4.
Summary of the ten-class classification problem.

Parameters for classification model
BS	16
LR	0.001
E	15
EP	6
SC	5
LF	BCE
OP	ADAM
Table 5.
Summary of parameters for classification experiments.

All of the studies were conducted on an Intel Xeon Processor E5–2697 v4, 2.3 GHz with sixty-four GB RAM and a sixteen GB NVIDIA GeForce GTX 1080 GPU using the PyTorch library and Python 3.7.

3.4 Performance matrix
Important performance metrics for classification experiment is stated in Eqs. (1)–(5):

E1
E2
E3
E4
E5
Here, true positive (TP) is the number of correctly classified healthy leaf images and true negative (TN) is the number of correctly classified unhealthy leaf images. False-positive (FP) and false-negative (FN) are the misclassified healthy and unhealthy leaf images, respectively.

Advertisement
4. Results
The performance of various networks in the different experiments is reported in this section.

In this study, three different experiments were conducted for tomato leaf images and the comparative performance for four different CNNs for the three classification schemes is shown in Table 6. It is apparent from Table 6 that all the evaluated pre-trained models perform very well in classifying healthy and unhealthy tomato leaf images in two-class, six-class, and ten-class problems.

Classification	Model	Overall	Weighted	
Binary Classification		Accuracy	Precision	Sensitivity	F1-score	Specificity
ResNet18	98.4	98.4	98.37	98.37	95.2
MobileNet	98.42	98.42	98.38	98.33	95.45
DenseNet201	98.9	98.85	98.66	98.76	95.56
Inceptionv3	99.2	99.23	99.2	99.25	96
Six-Class Classification	ResNet18	96.86	96.84	96.84	96.84	99.18
MobileNet	96.74	96.76	96.74	96.74	99.25
DenseNet201	97.99	97.99	97.99	97.98	99.54
Inceptionv3	97.65	97.67	97.65	97.63	99.41
Ten-Class Classification	ResNet18	96.75	96.77	96.79	96.76	99.65
MobileNet	97.2	97.18	97.19	97.17	99.7
DenseNet201	98.05	98.03	98.03	98.03	99.76
Inceptionv3	97.35	97.34	97.35	97.34	99.69
Table 6.
Summary of the tomato leaf disease classification performance using original leaf images.

Among the networks trained with leaf images for two-class, six-class, and ten-class problems, Densenet201 outperformed other trained models except without segmented two-class and with segmented six class problems where InceptionV3 was the best-performing network. Moreover, shallow networks ResNet18, and MobilenetV2 both showed comparable performance to most of the deep networks for the classification of images.

DenseNet201 outperforms others and for six-class and ten-class problems showed accuracy, sensitivity, and specificity of 97.99%, 97.99%, 99.54% and 98.05%, 98.03%, 99.76%, respectively. On the other hand, InceptionV3 produced the best result with accuracy, sensitivity, and specificity of 99.2%, 99.2%, and 96%, respectively for the two-class problem. Figure 3 clearly shows that the Receiver operating characteristic (ROC) curves for two-class, six-class, and ten-class problems of tomato leaf images. It is evident from Figure 3 that network performances are comparable for 2-, 6- and 10-class problems. However, deep networks can provide better performance gain for 6- and 10-class problems.


Figure 3.
Comparison of the ROC curves for healthy, and unhealthy tomato leaf image classification using CNN based models for two-class, six-class, and ten-class Classification.
The confusion matrix for the best performing networks for the different classification problems are shown in Figure 4. It can be noticed that even with the best performing network InceptionV3 for two-class tomato leaf images, 69 out of 16,570 unhealthy tomato leaf images were miss-classified as healthy and 74 out of 1,591 healthy tomato leaf images were miss-classified as unhealthy images.


Figure 4.
Confusion Matrix for healthy, and unhealthy tomato leaf image classification using CNN based models for (A) Binary-class, (B) six-class, and (C) ten-class Classification.
For the six-class problem, which consisted of one healthy class and five different unhealthy classes, only 27 out of 1,591 healthy tomato leaf images were miss-classified as unhealthy images, and 385 out of 16,570 unhealthy tomato leaf images consisted of one healthy class and nine different unhealthy classes, only 32out of 1,591 healthy tomato leaf images were miss-classified as unhealthy images and 382 out of 16,570 unhealthy tomato leaf images of nine different categories were miss-classified as healthy or any other unhealthy classes.

Advertisement
5. Discussion
Plant diseases are a major threat to global food security. Latest technologies need to be applied to the agriculture sector to curb diseases. Artificial intelligence-based technologies are extensively investigated in plant disease detection. Computer vision-based disease detection systems are popular for their robustness, ease of acquiring data, and quick results. This research investigates how different CNN-based architectures perform on classification of tomato leaf images. The study was divided into 3 sub-studies of 2 class classification (Healthy, and Unhealthy), 6 class classification (Healthy, Fungi, Bacteria, Mold, Virus, and Mite), and 10 class classification (Healthy, Early blight, Septoria leaf spot, Target spot, Leaf mold, Bacterial spot, Late bright mold, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, and Two-spotted spider mite). Overall, the DenseNet201 model outperformed every other model except for binary classification, where the InceptionV3 model outperformed other models. In the binary classification of healthy and diseased tomato leaves, InceptionV3 showed an overall accuracy of 99.2%, while DensNet201 showed an overall accuracy of 99.67%. In 6 class classification, DenseNet201 showed an overall accuracy of 97.99%, while InceptionV3 showed an overall accuracy of 97.65%. In 10 class classification, DenseNet201 showed an overall accuracy of 98.05%, while InceptionV3 showed an overall accuracy of 97.35%. The results in the paper are comparable to the state-of-the-art results and are also summarized in Table 7. Although the Plant Village dataset used in this study contains images taken in diverse environmental conditions, the dataset is collected in a specific region and is of specific breeds of tomatoes. A study conducted using a dataset containing images of other breeds of tomato plants from different regions of the world may result in a more robust framework for early disease detection in tomato plants. Furthermore, the lighter architecture of CNN models with non-linearity in the feature extraction layers might be useful to investigate for portable solutions.

Paper	Database	Reported performance
P. Tm et al. [59] (2018)	Plant village dataset (10 classes)	Accuracy-94.85%, Precision-94.81%, Sensitivity-94.81%, F1 Score − 94.81%
Mohit et al. [24] (2020)	Plant village dataset (10 classes)	Accuracy-91.20%, Precision-90.90%, Sensitivity-92.90%, F1 Score- 91.60%
H. Durmuş et al. [25](2017)	Plant village dataset (10 classes)	Accuracy-95.50%
Keke et al. [60] (2018)	Plant village dataset (2 classes)	Accuracy-97.20%
Belal et al. [61] (2018)	Plant village dataset (2 classes)	Accuracy-98.00%
Ouhami et al. [62] (2020)	Own dataset (6 classes, 666 images)	Accuracy-95.65%
Fuentes et al. [63] (2018)	Plant village dataset (9 classes)	Accuracy-96.00%
Madhavi et al. [64] (2020)	Own dataset (2 classes, 520 images)	Accuracy-80.00%
Proposed Study	Plant village dataset (2 classes,6 classes, and 10 classes) 18162 images	(Binary Class)
Accuracy-99.2%, Precision- 99.23%, Sensitivity-99.2%, F1 Score- 99.25%
(Six Class)
Accuracy-97.99%, Precision- 97.99%, Sensitivity-97.98%, F-1 Score- 97.54%
and
(Ten Class)
Accuracy-98.05%, Precision- 98.05%, Sensitivity-98.03%, F-1 Score- 98.03%
Table 7.
Results in the paper compared with other state of the art results.

Advertisement
6. Conclusion
The stages of the process into the infinite possibilities of machine learning for agriculture applications, complete with case studies. ResNet, MobileNet, DenseNet201, and InceptionV3 are examples of state-of-the-art pre-trained CNN models that do an excellent work of classifying diseases from plant leaf images. When compared to other architectures, the DenseNet201 was found to be better at extracting discriminative features from images. The trained models can be used to detect plant diseases early and automatically. As a result, preventive actions can be adopted faster. This research could help with early and automated disease detection in tomato crops, due to the use of cutting-edge technology like smartphones, drone cameras, and robotic platforms. The proposed structure can be combined with a feedback system that provides appropriate insights, treatments, disease prevention, and control techniques, resulting in improved crop yields.
