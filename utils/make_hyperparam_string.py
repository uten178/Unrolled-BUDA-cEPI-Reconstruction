import os


def make_hyperparam_string(model_type, 
                           data_consistency_type, 
                           data_consistency_before_reg, 
                           num_GD_blocks, 
                           num_inner_GD_iters, 
                           GD_step_size, 
                           learning_rate_base, 
                           batch_size, 
                           activation_type, 
                           INCLUDE_DATE=False):
    
    model_str = model_type + '_' + data_consistency_type
    if data_consistency_before_reg:
        model_str += '_DC_before_reg' 
    else:
        model_str += '_reg_before_DC'
    
    # Date and time
    if INCLUDE_DATE:
        import datetime
        now = datetime.datetime.now()
        hparam = str(now.year)
        if now.month < 10:
            hparam += "0" + str(now.month)
        else:
            hparam += str(now.month)

        if now.day < 10:
            hparam += "0" + str(now.day) + "_"
        else:
            hparam += str(now.day) + "_"
    else:
        hparam = ""

    # Hyper-parameters

    hparam += str(num_GD_blocks) + "blocks_" + str(num_inner_GD_iters) + 'inIter'+ str(GD_step_size) + 'stepSize_'+ str(learning_rate_base) + "lr_" + str(batch_size) + "batch_" + activation_type

    return os.path.join(model_str, hparam)
