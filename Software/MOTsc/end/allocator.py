
def allocate(stracks, dense_region):
    strks = []
    ftrks = []
    for trk in stracks:
        x = (trk.tlbr[2] + trk.tlbr[0]) / 2
        y = (trk.tlbr[3] + trk.tlbr[1]) / 2
        if (x > dense_region[0] and x < dense_region[2] and y > dense_region[1] and y < dense_region[3]):
            strks.append(trk)
        else:
            ftrks.append(trk)
    return strks, ftrks

def joint(strks, ftrks):
    exists = {}
    res = []
    for t in strks:
        exists[t.track_id] = 1
        res.append(t)
    for t in ftrks:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res