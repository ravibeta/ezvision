def stable_groups(keypoints, groups, threshold):
    for kp in keypoints:
        matched = false
        for group in groups:
            mean_feature = get_mean_feature(group)
            recent_pixel = get_recent_pixel(group)
            if kd.feature - mean_feature < threshold and abs(lucas_kanade_optical_flow(recent_pixel)-kp.pixel) < threshold:
               group.add(kp)
               matched = true
               break
        if matched == false:
            groups.add(create_group(kp))
def global_groups(stable_groups,global_groups, threshold):
    for stable_group in stable_groups:
        matched = false
        for global_group in global_groups:
            mean_feature = get_mean_feature(global_group)
            recent_pixel = get_recent_pixel(global_group)
            if get_mean_feature(stable_group) - mean_feature < threshold and delta_least_squares(stable_group,global_group):
               global_group.add(stable_group)
               matched = true
               break
        if matched == false:
            global_groups.add(create_global_group(stable_group))