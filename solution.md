    ### THIS IS WHERE WE IMPLEMENT ATTENTION ###
    ############################################
    # Let's try to do this as an exercice. The attention formula is:
    # (1) att = softmax((qT.k)/sqrt(dv))
    # (2) out = att.v   
    # We can use the functions K.softmax(), and K.batch_dot() for the dot product
    ############################################
    # Your turn! Compute att and out below...
    att = K.batch_dot(q,k ,axes=[-1,-1]) / np.sqrt(dv) #output_shape: (?, l, l)
    att = K.softmax(att,axis=2) #output_shape: (?, l, l)
    out = K.batch_dot(att,v, axes=[2,1]) #output_shape: (?, l, dv)
    ### END OF THE EXERCICE (don't touch the code below!)
    ############################################
