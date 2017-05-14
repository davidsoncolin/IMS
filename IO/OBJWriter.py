
def writeOBJ(filename, Geom_Dict):
    file = open(filename,'w')
    out_str = '# Units are in CM as opposed to the IMS MM\n'
    for v in Geom_Dict['v']:
        out_str += 'v {} {} {}\n'.format(v[0]/10, v[1]/10, v[2]/10)
    print "Vs written"
    for vn in Geom_Dict['vn']:
        out_str += 'vn {} {} {}\n'.format(vn[0], vn[1], vn[2])
    print "VNs written"
    for vt in Geom_Dict['vt']:
        out_str += 'vt {} {}\n'.format(vt[0], vt[1])
    print "VTs written"
    cur_index = 0
    for next_index in Geom_Dict['fs_splits']:
        f = Geom_Dict['fs'][cur_index:next_index]
        if len(f) > 0:
            out_str += 'f '
            for v in f:
                out_str += '/'.join(map(str,[v+1,v+1,v+1])) + ' '
            out_str += '\n'
        cur_index = next_index
    print "Fs written"
    file.write(out_str)
    file.close()
