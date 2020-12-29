#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "mpi_utilities.h"
#include "readubyte.h"
#include "CudaHelper.h"
#include "LayerBlock.h"
#include "Network.h"
#include "Activation.h"

#include <unistd.h>

int main(int argc,char **argv){
    DLMG dlmg(argc,argv);
    Network model(dlmg);

    /* ====================================================================== */
    //printf("Reading input data\n");

    // Read dataset sizes
    size_t width,height,channels = 1;
    size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(),FLAGS_train_labels.c_str(),nullptr,nullptr,width,height);
    size_t test_size  = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr,nullptr,width,height);
    if(train_size == 0) return 1;

    std::vector<uint8_t> train_images(train_size * width * height * channels),train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels),test_labels(test_size);

    // Read data from datasets
    size_t ts1 = ReadUByteDataset(FLAGS_train_images.c_str(),FLAGS_train_labels.c_str(),&train_images[0],&train_labels[0],width,height);
    size_t ts2 = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width,height);
    if(ts1 != train_size) return 2;
    if(ts2 != test_size) return 3;

    //printf("Done. Training dataset size: %d, Test dataset size: %d\n",(int)train_size,(int)test_size);
    //printf("Batch size: %lld, iterations: %d\n",FLAGS_batch_size,FLAGS_iterations);

    /* ---------------------------------------------------------------------- */
    //printf("Preparing Testing Dataset\n");

    // Normalize training set to be in [0,1]
    std::vector<Real> train_images_Real(train_images.size());
    for (size_t i = 0; i < train_size * channels * width * height; ++i) {
        train_images_Real[i] = (Real)train_images[i] / 255.0;
    }

    std::vector<Real> train_labels_Real(train_size);
    for (size_t i = 0; i < train_size; ++i) {
        train_labels_Real[i] = (Real)train_labels[i];
    }

    // Normalize image to be in [0,1]
    std::vector<Real> test_images_Real(test_images.size());
    for (size_t i = 0; i < test_size * channels * width * height; ++i) {
        test_images_Real[i] = (Real)test_images[i] / 255.0;
    }
    /* ====================================================================== */

   /** std::random_device rd_image;
     * int random_image = rd_image() % train_size;
     * std::stringstream ss; ss << "image-" << (int)train_labels[random_image] << ".pgm";
     * SavePGMFile(&train_images[0] + random_image * width*height*channels,width,height,ss.str().c_str());
     */

    /* ---------------------------------------------------------------------- */
    /* ========================== */
    /* Build Network Architecture */
    /* ========================== */
    Real dt = 0.001;

//    // Create the LeNet network architecture
//    ConvolutionLayer conv1((int) FLAGS_batch_size,
//                           (int) channels,
//                           (int) width,
//                           (int) height,
//      /*  kernel size = */ 5,
//      /* out_channels = */ 20,
//           /* padding = */ 1,
//                           RELU);
//
//    MaxPoolLayer pool1((int) FLAGS_batch_size,
//                       conv1.out_channels,
//                       conv1.out_width,
//                       conv1.out_height,
//                       2,2);
//
//    ConvolutionLayer conv2((int) FLAGS_batch_size,
//                           pool1.out_channels,
//                           pool1.out_width,
//                           pool1.out_height,
//      /*  kernel size = */ 5,
//      /* out_channels = */ 50,
//           /* padding = */ 1,
//                           RELU);
//
//    /* residual convolution layer */
//    ConvolutionLayer conv3((int) FLAGS_batch_size,
//                           conv2.out_channels,
//                           conv2.out_width,
//                           conv2.out_height,
//      /*  kernel size = */ 3,
//                           RELU,
//                           dt);
//
//    MaxPoolLayer pool2((int) FLAGS_batch_size,
//                       conv3.out_channels,
//                       conv3.out_width,
//                       conv3.out_height,
//                       2,2);
//
//    FullyConnectedLayer fc1((int) FLAGS_batch_size,
//                            pool2.getOutSize(),
//                            512,
//                            RELU);
//
//    FullyConnectedLayer fc2((int) FLAGS_batch_size,
//                            fc1.getOutSize(),
//                            RELU,
//                            dt);
//
//    /* residual fully connected layer */
//    FullyConnectedLayer fc3((int) FLAGS_batch_size,
//                            fc2.getOutSize(),
//                            10,
//                            IDENTITY);


    // Create the LeNet network architecture
    ConvolutionLayer conv0((int) FLAGS_batch_size,
                           (int) channels,
                           (int) width,
                           (int) height,
      /*  kernel size = */ 7,
      /* out_channels = */ 20,
           /* padding = */ 1,
                           RELU);
    ConvolutionLayer conv1((int) FLAGS_batch_size, conv0.out_channels, conv0.out_width, conv0.out_height, 7, RELU, dt);

    int nconv = 256;
    ConvolutionLayer **conv_layers = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers2 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers3 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers4 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers5 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers6 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers7 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers8 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers9 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers10 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers11= (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers12 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers13 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers14 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers15 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));
//    ConvolutionLayer **conv_layers16 = (ConvolutionLayer **) malloc(nconv*sizeof(ConvolutionLayer *));

    for(int i = 0; i < nconv; ++i){
        conv_layers[i] = new ConvolutionLayer((int) FLAGS_batch_size,
                                              conv1.out_channels,
                                              conv1.out_width,
                                              conv1.out_height,
                         /*  kernel size = */ 7,
                                              RELU,
                                              dt);
//        conv_layers2[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers3[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers4[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers5[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers6[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers7[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers8[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers9[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers10[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers11[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers12[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers13[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers14[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers15[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
//        conv_layers16[i] = new ConvolutionLayer((int) FLAGS_batch_size,conv1.out_channels,conv1.out_width,conv1.out_height,7,RELU,dt);
    }

    FullyConnectedLayer fc1((int) FLAGS_batch_size, conv1.getOutSize(), 10, IDENTITY);
//    FullyConnectedLayer fc2((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc3((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc4((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc5((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc6((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc7((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc8((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc9((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc10((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc11((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc12((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc13((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc14((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc15((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
//    FullyConnectedLayer fc16((int) FLAGS_batch_size, conv1.getOutSize(), conv1.getOutSize(), IDENTITY);
    SoftmaxLayer softmax((int) FLAGS_batch_size, fc1.getOutSize());

    /* add all layers to network */
    model.add(&conv0);
    model.add(&conv1);
    for(int i = 0; i < nconv; ++i) model.add(conv_layers[i]);
//    model.add(&fc2); for(int i = 0; i < nconv; ++i) model.add(conv_layers2[i]);
//    model.add(&fc3); for(int i = 0; i < nconv; ++i) model.add(conv_layers3[i]);
//    model.add(&fc4); for(int i = 0; i < nconv; ++i) model.add(conv_layers4[i]);
//    model.add(&fc5); for(int i = 0; i < nconv; ++i) model.add(conv_layers5[i]);
//    model.add(&fc6); for(int i = 0; i < nconv; ++i) model.add(conv_layers6[i]);
//    model.add(&fc7); for(int i = 0; i < nconv; ++i) model.add(conv_layers7[i]);
//    model.add(&fc8); for(int i = 0; i < nconv; ++i) model.add(conv_layers8[i]);
//    model.add(&fc9); for(int i = 0; i < nconv; ++i) model.add(conv_layers9[i]);
//    model.add(&fc10); for(int i = 0; i < nconv; ++i) model.add(conv_layers10[i]);
//    model.add(&fc11); for(int i = 0; i < nconv; ++i) model.add(conv_layers11[i]);
//    model.add(&fc12); for(int i = 0; i < nconv; ++i) model.add(conv_layers12[i]);
//    model.add(&fc13); for(int i = 0; i < nconv; ++i) model.add(conv_layers13[i]);
//    model.add(&fc14); for(int i = 0; i < nconv; ++i) model.add(conv_layers14[i]);
//    model.add(&fc15); for(int i = 0; i < nconv; ++i) model.add(conv_layers15[i]);
//    model.add(&fc16); for(int i = 0; i < nconv; ++i) model.add(conv_layers16[i]);
    model.add(&fc1);
    model.add(&softmax);

    model.initialize(FLAGS_batch_size);
    /* ---------------------------------------------------------------------- */

    /* ---------------------------------------------------------------------- */
    /* ================================ */
    /* Initialize Training Context Data */
    /* ================================ */
    if (!FLAGS_pretrained) {
        std::random_device rd;
        std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

        model.parametersInitializeHost(gen);
    }

    /* Copy initial network to device */
    model.parametersHostToDeviceAsync(0);
    /* ---------------------------------------------------------------------- */

    /* ====================================================================== */
    /* Train Network */
    dataset_t dataset;
    dataset.test_accuracy_interval = 1;
    dataset.train_size = train_size;
    dataset.test_size = test_size;
    dataset.channels = channels;
    dataset.height = height;
    dataset.width = width;

    dataset.h_train_data = train_images_Real.data();
    dataset.h_train_labels = train_labels_Real.data();
    dataset.h_test_data = test_images_Real.data();
    dataset.h_test_labels = test_labels.data();

    int nepochs = 1;
    model.fit(&dataset, FLAGS_batch_size, nepochs);

    /* Classification Error Testing */
    model.evaluate(&dataset, FLAGS_classify, "test");
    /* ====================================================================== */

    for(int i = 0; i < nconv; ++i) delete conv_layers[i]; free(conv_layers);
    return 0;
}