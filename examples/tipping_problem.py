import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as mpl
# Step 1 : Fuzzify Input / Output 
#Input  Variables have the same domain/ Universe functions
service = np.arange(0,11,1)
food    = np.arange(0,11,1)

# Output Variables Domain
tip = np.arange(0,101,1) 

# Input Membership Functions
# Service
ser_p = fuzz.gaussmf(service ,0,1.5)
ser_g = fuzz.gaussmf(service ,5,1.5)
ser_e = fuzz.gaussmf(service ,10,1.5)

# Food
foo_r = fuzz.trapmf(food , [0, 0, 1, 3])
foo_d = fuzz.gaussmf(food, 10,1.5)

# Output tip
tip_ch  = fuzz.trimf(tip, [0, 15, 30])
tip_ave = fuzz.trimf(tip, [25, 45, 70])
tip_gen = fuzz.trimf(tip, [60, 80, 100])
print 'tip_cheap', tip_ch
print 'tip_average', tip_ave
print 'tip_generous', tip_gen

# Here I'll use your example service == 2 and food == 4
# These return a single value, which is combined using the rules in Step 3 
# and operates on tip membership functions in Step 4
food_1 = fuzz.interp_membership(food, foo_r, 4.)
food_3 = fuzz.interp_membership(food, foo_r, 4.)
service_1 = fuzz.interp_membership(service, ser_p, 2)
service_2 = fuzz.interp_membership(service, ser_g, 2)
service_3 = fuzz.interp_membership(service, ser_e, 2)
print 'food_1', food_1
print 'food_3', food_3
print 'service_1', service_1
print 'service_2', service_2
print 'service_3', service_3

# First rule is OR - this is a max operator
rule1 = np.fmax(food_1, service_1)  # Doable with inbuilt Python max(), but np.fmax is more general
rule2 = service_2  # No combination, this is just passed
rule3 = np.fmax(food_3, service_3)
print 'rule1', rule1
print 'rule2', rule2
print 'rule3', rule3


# Product is simple multiplication of weight w/fuzzy membership function
# min is np.fmin(weight, membership)
# Here I use product because that appears to be your preference
imp1 = rule1 * tip_ch
imp2 = rule2 * tip_ave
imp3 = rule3 * tip_gen

print 'imp1', imp1
print 'imp2', imp2
print 'imp3', imp3

# Aggregate
aggregate_membership = np.fmax(imp1, np.fmax(imp2, imp3))
mpl.plot(aggregate_membership)
mpl.show()

# Defuzzify
tip = fuzz.defuzz(tip, aggregate_membership, 'centroid')
print tip # correct answer is
print np.allclose(tip, 19.4, atol=0.05)
