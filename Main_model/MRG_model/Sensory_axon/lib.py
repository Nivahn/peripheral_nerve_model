from neuron import h
import matplotlib.pyplot as plt
import numpy as np

def mrg_params(fiberD=10.0):
    if fiberD not in MRG_TABLE:
        raise ValueError(f"fiberD {fiberD} не в таблице.")
    g, axonD, nodeD, paraD1, paraD2, deltax, paralength2, nl = MRG_TABLE[fiberD]
    interlength = (deltax - nodelength - 2*paralength1 - 2*paralength2)/6.0
    Rpn0 = rin_peri(rhoa, nodeD,  space_p1)
    Rpn1 = rin_peri(rhoa, paraD1, space_p1)
    Rpn2 = rin_peri(rhoa, paraD2, space_p2)
    Rpx  = rin_peri(rhoa, axonD,  space_i)
    Lstep = 2*paralength1 + 2*paralength2 + 6*interlength + nodelength
    return dict(fiberD=fiberD, axonD=axonD, nodeD=nodeD,
                paraD1=paraD1, paraD2=paraD2,
                paral1=paralength1, paral2=paralength2,
                interL=interlength, nl=nl,
                Rpn0=Rpn0, Rpn1=Rpn1, Rpn2=Rpn2, Rpx=Rpx, Lstep=Lstep)



def rin_peri(rhoa_ohmum, inner_d_um, gap_um):
    """Продольное сопротивление периакисонального пространства для extracellular.xraxial."""
    return (rhoa_ohmum*0.01)/(math.pi*(((inner_d_um/2+gap_um)**2) - (inner_d_um/2)**2))

def ins(sec, mech: str):
    """Вставляет механизм, если его ещё нет."""
    if int(h.ismembrane(mech, sec=sec)) == 0:
        sec.insert(mech)

def set_extr(sec, xraxial, xg, xc):
    """extracellular-параметры: обязательно с индексом [0]."""
    ins(sec, 'extracellular')
    for seg in sec:
        seg.xraxial[0] = xraxial
        seg.xg[0]      = xg
        seg.xc[0]      = xc


def pick_node_mech():
    """Автовыбор доступного нодального механизма."""
    tmp = h.Section()
    try:
        tmp.insert('newaxnode'); tmp.uninsert('newaxnode')
        return 'newaxnode'
    except:
        pass
    try:
        tmp.insert('axnode'); tmp.uninsert('axnode')
        return 'axnode'
    except:
        pass
    raise RuntimeError("Не найден ни 'newaxnode', ни 'axnode' — скомпилируйте .mod.")

NODE_MECH = pick_node_mech()



# ---------- КОНСТРУКТОРЫ СЕКЦИЙ ----------
def make_node(nodeD, nodel, rhoa, Rpn0, node_mech=None,
              gnabar=None, gnapbar=None, el=None):
    global _node_id
    s = h.Section(name=f'node_{_node_id}'); _node_id += 1
    s.nseg = 1
    s.L    = nodel
    s.diam = nodeD
    s.Ra   = rhoa/10000.0
    s.cm   = 2.0
    mech   = node_mech or NODE_MECH
    ins(s, mech)
    if mech == 'newaxnode':
        s.el_newaxnode      = -90.0 if el is None else el
        s.gnabar_newaxnode  = 3.0   if gnabar  is None else gnabar
        s.gnapbar_newaxnode = 0.005 if gnapbar is None else gnapbar
    set_extr(s, Rpn0, 1e10, 0.0)
    REG["node"].append(s)
    return s

def make_mysa(fiberD, paraD1, paral1, rhoa, mygm, mycm, nl, Rpn1):
    global _mysa_id
    s = h.Section(name=f'MYSA_{_mysa_id}'); _mysa_id += 1
    s.nseg = 1
    s.L    = paral1
    s.diam = fiberD
    ratio  = paraD1/fiberD
    s.Ra   = rhoa*(1.0/(ratio*ratio))/10000.0
    s.cm   = 2.0*ratio
    ins(s, 'pas'); s.g_pas = 0.001*ratio; s.e_pas = -80.0
    set_extr(s, Rpn1, mygm/(nl*2.0), mycm/(nl*2.0))
    REG["mysa"].append(s)
    return s

def make_flut(fiberD, paraD2, paral2, rhoa, mygm, mycm, nl, Rpn2):
    global _flut_id
    s = h.Section(name=f'FLUT_{_flut_id}'); _flut_id += 1
    s.nseg = 1
    s.L    = paral2
    s.diam = fiberD
    ratio  = paraD2/fiberD
    s.Ra   = rhoa*(1.0/(ratio*ratio))/10000.0
    s.cm   = 2.0*ratio
    ins(s, 'pas'); s.g_pas = 0.0001*ratio; s.e_pas = -80.0
    set_extr(s, Rpn2, mygm/(nl*2.0), mycm/(nl*2.0))
    REG["flut"].append(s)
    return s

def make_stin(fiberD, axonD, interL, rhoa, mygm, mycm, nl, Rpx):
    global _stin_id
    s = h.Section(name=f'STIN_{_stin_id}'); _stin_id += 1
    s.nseg = 1
    s.L    = interL
    s.diam = fiberD
    ratio  = axonD/fiberD
    s.Ra   = rhoa*(1.0/(ratio*ratio))/10000.0
    s.cm   = 2.0*ratio
    ins(s, 'pas'); s.g_pas = 0.0001*ratio; s.e_pas = -80.0
    set_extr(s, Rpx, mygm/(nl*2.0), mycm/(nl*2.0))
    REG["stin"].append(s)
    return s

# ---------- ОДИН ШАГ MRG (между узлами): MYSA→FLUT→STIN×6→FLUT→MYSA→node ----------
def append_one_step(parent_node, P, node_mech=None):
    mysa0 = make_mysa(P['fiberD'], P['paraD1'], P['paral1'], rhoa, mygm, mycm, P['nl'], P['Rpn1'])
    flut0 = make_flut(P['fiberD'], P['paraD2'], P['paral2'], rhoa, mygm, mycm, P['nl'], P['Rpn2'])
    st = [make_stin(P['fiberD'], P['axonD'], P['interL'], rhoa, mygm, mycm, P['nl'], P['Rpx']) for _ in range(6)]
    flut1 = make_flut(P['fiberD'], P['paraD2'], P['paral2'], rhoa, mygm, mycm, P['nl'], P['Rpn2'])
    mysa1 = make_mysa(P['fiberD'], P['paraD1'], P['paral1'], rhoa, mygm, mycm, P['nl'], P['Rpn1'])
    nxt   = make_node(P['nodeD'], nodelength, rhoa, P['Rpn0'], node_mech=node_mech or NODE_MECH)

    # топология
    mysa0.connect(parent_node, 1.0, 0.0)
    flut0.connect(mysa0,       1.0, 0.0)
    st[0].connect(flut0,       1.0, 0.0)
    for k in range(1,6):
        st[k].connect(st[k-1], 1.0, 0.0)
    flut1.connect(st[5],       1.0, 0.0)
    mysa1.connect(flut1,       1.0, 0.0)
    nxt.connect(mysa1,         1.0, 0.0)
    return nxt

def build_chain(n_nodes, P, node_mech=None):
    nodes = [make_node(P['nodeD'], nodelength, rhoa, P['Rpn0'], node_mech=node_mech or NODE_MECH)]
    for _ in range(n_nodes-1):
        nxt = append_one_step(nodes[-1], P, node_mech=node_mech or NODE_MECH)
        nodes.append(nxt)
    return nodes


# ---------- ОДИН ШАГ MRG (между узлами): MYSA→FLUT→STIN×6→FLUT→MYSA→node ----------
def append_one_step(parent_node, P, node_mech=None):
    mysa0 = make_mysa(P['fiberD'], P['paraD1'], P['paral1'], rhoa, mygm, mycm, P['nl'], P['Rpn1'])
    flut0 = make_flut(P['fiberD'], P['paraD2'], P['paral2'], rhoa, mygm, mycm, P['nl'], P['Rpn2'])
    st = [make_stin(P['fiberD'], P['axonD'], P['interL'], rhoa, mygm, mycm, P['nl'], P['Rpx']) for _ in range(6)]
    flut1 = make_flut(P['fiberD'], P['paraD2'], P['paral2'], rhoa, mygm, mycm, P['nl'], P['Rpn2'])
    mysa1 = make_mysa(P['fiberD'], P['paraD1'], P['paral1'], rhoa, mygm, mycm, P['nl'], P['Rpn1'])
    nxt = make_node(P['nodeD'], nodelength, rhoa, P['Rpn0'], node_mech=node_mech or NODE_MECH)

    # топология
    mysa0.connect(parent_node, 1.0, 0.0)
    flut0.connect(mysa0, 1.0, 0.0)
    st[0].connect(flut0, 1.0, 0.0)
    for k in range(1, 6):
        st[k].connect(st[k - 1], 1.0, 0.0)
    flut1.connect(st[5], 1.0, 0.0)
    mysa1.connect(flut1, 1.0, 0.0)
    nxt.connect(mysa1, 1.0, 0.0)

    return nxt


def build_chain(n_nodes, P, node_mech=None):
    nodes = [make_node(P['nodeD'], nodelength, rhoa, P['Rpn0'], node_mech=node_mech or NODE_MECH)]
    for _ in range(n_nodes - 1):
        nxt = append_one_step(nodes[-1], P, node_mech=node_mech or NODE_MECH)
        nodes.append(nxt)
    return nodes

def scaled_params(P, diam_scale=0.6):
    """Сужение диаметров после ветвления."""
    out = dict(P)
    out['fiberD'] *= diam_scale
    out['axonD']  *= diam_scale
    out['nodeD']  *= diam_scale
    out['paraD1'] *= diam_scale
    out['paraD2'] *= diam_scale
    # длины оставляем как есть; при желании можно тоже масштабировать
    out['Rpn0'] = rin_peri(rhoa, out['nodeD'],  space_p1)
    out['Rpn1'] = rin_peri(rhoa, out['paraD1'], space_p1)
    out['Rpn2'] = rin_peri(rhoa, out['paraD2'], space_p2)
    out['Rpx']  = rin_peri(rhoa, out['axonD'],  space_i)
    # Lstep не меняем, если длины не масштабируем
    return out

def check_branching():
    for sec in h.allsec():
        parent = sec.parentseg()
        if parent:
            parent_sec = parent.sec
            children = []
            for child in h.allsec():
                if hasattr(child, 'parentseg') and child.parentseg():
                    if child.parentseg().sec == sec:
                        children.append(child.name())
            if len(children) > 1:
                print(f"ВЕТВЛЕНИЕ: {sec.name()} -> {children}")


def reset_model():
    # удалить все секции из ядра NEURON
    h('forall delete_section()')
    # обнулить наши реестры и счётчики
    REG["node"].clear(); REG["mysa"].clear(); REG["flut"].clear(); REG["stin"].clear()
    global _node_id, _mysa_id, _flut_id, _stin_id
    _node_id = _mysa_id = _flut_id = _stin_id = 0