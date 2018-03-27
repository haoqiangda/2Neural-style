from ns_layers import *

style_transfer('styles/tubingen.jpg','styles/composition_vii.jpg',192,512,3,(1, 4, 6, 7),5e-2,(20000, 500, 12, 1),5e-2)
# def content_loss_test(correct):
#     content_layer = 3
#     content_weight = 6e-2
#     c_feats = sess.run(model.extract_features()[content_layer], {model.image: content_img_test})
#     bad_img = tf.zeros(content_img_test.shape)
#     feats = model.extract_features(bad_img)[content_layer]
#     student_output = sess.run(content_loss(content_weight, c_feats, feats))
#     error = rel_error(correct, student_output)
#     print('Maximum error is {:.3f}'.format(error))

# content_loss_test(answers['cl_out'])