import collections


def calculate_congestion(datas, frame):
    # 80 percent of summary_frame
    criterion = int(0.8 * frame)

    # Count id values
    count_person = dict(collections.Counter(datas))
    count_standing = 0

    for i in count_person.values():
        if i > criterion:
            count_standing += 1

    if count_standing < 14:
        level = 'Spare'
    elif count_standing < 17:
        level = 'General'
    elif count_standing < 22:
        level = 'Caution'
    else:
        level = 'Congestion'

    return level, count_standing
