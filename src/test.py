import src.model as model

image_shape_coarse=(64,64,64,4)
mask_shape_coarse=(64,64,64,1)
image_shape_x_coarse=(32,32,32,64)
image_shape_fine=(64,64,64, 4)
mask_shape_fine=(64,64,64,1)
label_shape_fine = (64,64,64,1)
label_shape_coarse = (64,64,64,1)

g_model_coarse = model.coarse_generator(img_shape=(64,64,64,4),
                           mask_shape=(64,64,64,1),
                           ncf=64, n_downsampling=2,
                           n_blocks=9, n_channels=1)

g_model_fine=model.fine_generator(x_coarse_shape=(32,32,32,64),
                        input_shape=(64,64,64, 4),
                        mask_shape=(64,64,64,1),
                        nff=64, n_blocks=3,
                        n_coarse_gen=1,n_channels = 1)

d_model1 = model.discriminator_ae(input_shape_fundus=(64,64,64, 4),
                            input_shape_label=(64,64,64, 1),
                            ndf=32, n_layers=3, activation='tanh',
                            name='Discriminator1')

d_model2 = model.discriminator_ae(input_shape_fundus=(64,64,64, 4),
                            input_shape_label=(64,64,64, 1),
                            ndf=32, n_layers=3, activation='tanh',
                            name='Discriminator2')


RVGAN = model.RVgan(g_model_fine,
                    g_model_coarse,
                    d_model1,
                    d_model2,
                    image_shape_fine,
                    image_shape_coarse,
                    image_shape_x_coarse,
                    mask_shape_fine,
                    mask_shape_coarse,
                    label_shape_fine,
                    label_shape_coarse)