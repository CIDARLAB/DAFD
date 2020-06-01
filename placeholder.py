def run_analysis(features, tolerance, di):
    tol = tolerance/100 # Assuming that tolerance is given in percent, format
    feat_denorm = denormalize_features(features)
    max_feat, min_feat = make_tol_dicts(feat_denorm, tol)
    combos = all_combos(feat_denorm, min_feat, max_feat)
    combos_normed = [renormalize_features(option) for option in combos]
    start = t.time()
    outputs = [di.runForward(option) for option in combos_normed]
    e1 = t.time()
    print(e1-start)
    return outputs

def random_features(di):
    headers = di.MH.get_instance().input_headers
    ranges = di.ranges_dict
    feature_set = {head: (round(r.random()*(ranges[head][1] - ranges[head][0])+ranges[head][0], 2)) for head in headers}
    return feature_set

def all_combos(features, min_feat, max_feat):
    feat_op = []
    for key in features.keys():
        feat_op.append(
            [min_feat[key], features[key], max_feat[key]]
        )
    combo_Iter = itertools.product(feat_op[0], feat_op[1], feat_op[2], feat_op[3],
                               feat_op[4], feat_op[5], feat_op[6], feat_op[7])
    combos = []
    for option in combo_Iter:
        combos.append({key:option[i] for i,key in enumerate(features.keys())})
    return combos
