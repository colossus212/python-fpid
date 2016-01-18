#!/usr/bin/env python

# Modules
import json
import matplotlib.pyplot as mpl
import numpy as np
from skfuzzy import interp_membership, defuzz
from skfuzzy.membership import (gaussmf, gauss2mf, gbellmf, piecemf, pimf,
                                psigmf, sigmf, smf, trapmf, trimf, zmf)
import operator

class FPID:
    def __init__(self, C, P=None, I=None, D=None):

        # Proportional
        if P:
            self.p_range = np.arange(*P['range']) #!TODO
            self.p_mf = self.generate_mf_group(P['memberships'], self.p_range)
        else:
            self.p_range = None
            
        # Integral
        if I:
            self.i_range = np.arange(*I['range']) #!TODO
            self.i_mf = self.generate_mf_group(I['memberships'], self.i_range)
        else:
            self.i_range = None
            
        # Differential 
        if D:
            self.d_range = np.arange(*D['range']) #!TODO
            self.d_mf = self.generate_mf_group(D['memberships'], self.d_range)
        else:
            self.d_range = None
            
        # Fuzzy Class
        self.c_range = np.arange(*C['range']) #!TODO
        self.c_mf = self.generate_mf_group(C['memberships'], self.c_range)
        self.c_rules = C['rules']

    def generate_mf_group(self, G, x):
        mf_group = {}
        for (k, v) in G.iteritems():
            shp = v['shp']
            mf = v['mf']
            if mf == 'trap':
                mf_group[k] = trapmf(x, shp)
            if mf == 'tri':
                mf_group[k] = trimf(x, shp)
            if mf == 'gbell':
                mf_group[k] = gbellmf(x, shp[0], shp[1], shp[2])
            if mf == 'gauss':
                mf_group[k] = gaussmf(x, shp[0], shp[1])
            if mf == 'gauss2':
                mf_group[k] = gauss2mf(x, shp[0], shp[1])
            if mf == 'sig':
                mf_group[k] = sigmf(x, shp[0], shp[1])
            if mf == 'psig':
                mf_group[k] = psigmf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'zmf':
                mf_group[k] = zmf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'smf':
                mf_group[k] = smf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'pimf':
                mf_group[k] = pimf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'piecemf':
                mf_group[k] = piecemf(x, shp[0], shp[1], shp[2], shp[3])
        return mf_group
        
    def classify(self, p, i, d):
        
        # Calculate membership value for each function 
        if self.p_range is not None:
            p_interp = {k: interp_membership(self.p_range, mf, p) for k, mf in self.p_mf.iteritems()}
            print max(p_interp.iteritems(), key=operator.itemgetter(1))[0]
        else:
            p_interp = {}
        if self.i_range is not None:
            i_interp = {k: interp_membership(self.i_range, mf, i) for k, mf in self.i_mf.iteritems()}
            print max(i_interp.iteritems(), key=operator.itemgetter(1))[0]
        else:
            i_interp = {}
        if self.d_range is not None:
            d_interp = {k: interp_membership(self.d_range, mf, d) for k, mf in self.d_mf.iteritems()}
            print max(d_interp.iteritems(), key=operator.itemgetter(1))[0]
        else:
            d_interp = {}
        
        # Merge rule-bases
        dicts = [p_interp, i_interp, d_interp]
        super_dict = {}
        for k in set(k for d in dicts for k in d):
            super_dict[k] = [d[k] for d in dicts if k in d]

        # Generated inferences by rule implications
        aggregate_membership = np.zeros(len(self.c_range))
        for a,b,c in self.c_rules:
            try:
                impl = np.fmin(super_dict[a], super_dict[b]) * self.c_mf[c]
                aggregate_membership = np.fmax(impl, aggregate_membership)
            except:
                pass
        mpl.plot(aggregate_membership)
        mpl.show()
        c = defuzz(self.c_range, aggregate_membership, 'centroid')
        return c # this is the resulting "value" of the current state
        
    def show_mf_groups(self):
        if self.p_range is not None:
            mpl.subplot(4,1,1)
            for label, mf in self.p_mf.iteritems():
                mpl.plot(mf)
        if self.i_range is not None:
            mpl.subplot(4,1,2)
            for label, mf in self.i_mf.iteritems():
                mpl.plot(mf)
        if self.d_range is not None:
            mpl.subplot(4,1,3)
            for label, mf in self.d_mf.iteritems():
                mpl.plot(mf)
        if self.c_range is not None:
            mpl.subplot(4,1,4)
            for label, mf in self.c_mf.iteritems():
                mpl.plot(mf)
        mpl.show()
                
if __name__ == '__main__':
    with open('rules.json', 'r') as jsonfile:
        config = json.loads(jsonfile.read())
        ctrl = FPID(config['C'], P=config['P'], D=config['D'])
        #for p in range(-40,40):
        #    print '%d,0,0 --> %f' % (p, ctrl.classify(p, 0, 0))
        #for i in range(-40,40):
        #    print '10,%d,0 --> %f' % (i, ctrl.classify(10, i, 0))
        #for d in range(-5,5):
        #    print '10,20,%d --> %f' % (d, ctrl.classify(10, 20, d))
        ctrl.show_mf_groups()
        a,b,c = 10, 0, -5
        z = ctrl.classify(a,b,c)
        print a, b, c, z
