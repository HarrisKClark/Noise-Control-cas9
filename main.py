import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import random
import math
import copy

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams.update({'font.size': 20})

P0 = 1
T0 = 0.25
G0 = 0.5
h0 = 1
cx = 1
del_t = 0.1
del_m = 0.2
gamma = 0.3
L_t = 200 # see u(t)
L_c = 150 # same as n_x (length of coding region)
beta = 10
cf = 1
cr = 1
cn = 10

c1 = P0*cx*beta
c2 = del_t*T0*L_c*cx*P0/(2*h0*L_t)*beta
c3 = del_t*T0
c4 = gamma*G0
cf = 18
cr = 2
constants = [c1, c2, c3, c4, cf, cr, del_m]

sig = [1]
mRNA = [0]
state = [0]
t1 = [0]
t2 = [0]
t3 = [0]
params = [sig,mRNA,state, t1, t2, t3]

time = 5000

class agent:
    def __init__(self, constants, params, time, control):
        self.constants = constants
        self.params = params
        self.time = time
        self.control = control

    def kcat(self, sig):
        sig_star = -1
        s = 2
        return 2.71828 ** (-1 * ((sig - sig_star) ** 2) / s)

    def simulate(self):
        results = copy.deepcopy(self.params)

        c1 = self.constants[0]
        c2 = self.constants[1]
        c3 = self.constants[2]
        c4 = self.constants[3]
        cf = self.constants[4]
        cr = self.constants[5]
        del_m = self.constants[6]


        while (t1[-1] < self.time) and (t2[-1] < self.time) and (t3[-1] < self.time):

            current_sig = results[0][-1]
            current_mRNA = results[1][-1]
            current_state = results[2][-1]
            current_t1 = results[3][-1]
            current_t2 = results[4][-1]
            current_t3 = results[5][-1]

            #r1 mrna, r2 supercoiling, r3 state


            r1, r2, r3 = False, False, False

            times = (current_t1, current_t2, current_t3)
            m = min(times)
            x = [i for i, j in enumerate(times) if j == m]

            for i in x:
                if i == 0:
                    r1 = True
                if i == 1:
                    r2 = True
                if i == 2:
                    r3 = True

            #mrna
            if r1:
                rate1 = [c1 * self.kcat(current_sig), del_m * current_mRNA]

                abs_rates = []

                for rate in rate1:
                    abs_rates.append(abs(rate))

                rate1sum = sum(abs_rates)

                tau1 = np.random.exponential(scale=(1 / rate1sum))

                if (current_t1 + tau1) > self.time:
                    break

                results[3].append(current_t1 + tau1)

                rand = random.uniform(0, 1)

                # event 1 (mrna production)
                if rand * rate1sum <= abs_rates[0]:
                    results[1].append(current_mRNA + 0.1)

                # event 2 (mrna degredation)
                elif rand * rate1sum <= abs_rates[0] + abs_rates[1]:
                    results[1].append(current_mRNA - 0.1)

            #supercoiling
            if r2:
                rate2 = [c2 * self.kcat(current_sig), c3 * current_sig, c4 * current_sig]

                abs_rates = []

                for rate in rate2:
                    abs_rates.append(abs(rate))

                rate2sum = sum(abs_rates)

                tau2 = np.random.exponential(scale=(1 / rate2sum))
                results[4].append(current_t2 + tau2)

                rand = random.uniform(0, 1)

                # event 3 (transcirption induced supercoiling)
                if rand * rate2sum <= abs_rates[0]:
                    results[0].append(current_sig + 0.01 * current_state) #current state (eg. 0 unbound... no supercoils trapped)

                # event 4 (topoiomerase event)
                elif rand * rate2sum <= abs_rates[0] + abs_rates[1]:
                    results[0].append(current_sig + 0.005)

                # event 5 (gyrase event)
                elif rand * rate2sum <= abs_rates[0] + abs_rates[1] + abs_rates[2]:
                    results[0].append(current_sig - 0.005)

            #crispr bound
            if r3:
                #-1, neg  cf<-> cr 0 unbound, 1 pos

                rate3 = [cf, cr]
                rate3sum = sum(rate3)

                tau3 = np.random.exponential(scale=(1 / rate3sum))

                if (current_t3 + tau3) > self.time:
                    break

                results[5].append(current_t3 + tau3)

                rand = random.uniform(0, 1)

                #event 1 bind
                if rand * rate3sum <= rate3[0]:
                    if current_state == 0: # if unbound
                        results[2].append(1) # pos bound

                # event 2 unbind
                else:
                    results[2].append(0)

                if self.control == False:
                    results[2][-1] = 0

        return results

print("running")

agents = []
results = []

count = 20
for i in range(count):
    agents.append(agent(constants, params, time, True))

iiter = 1
for agent in agents:
    print("simulation " + str(iiter) + "/" + str(count))
    results.append(agent.simulate())
    iiter +=1

print("simulation done")
print("data processing")

interpreted = [[0]*1 for i in range(1, 490)]

iiter = 1
for result in results:
    print("processing " + str(iiter) + "/" + str(count))
    for i in range(1, 490):
        val1, val2 = 0, 0
        found = False
        for j in range(0, len(result[3])-1):
            if result[3][j] >= (10*i):
                val1,val2 = j-1, j
                found = True
            if found:
                break
        interpreted_value = result[1][val1] + ((result[1][val2] - result[1][val1])/(result[3][val2] - result[3][val1]))*(i*10 - result[3][val1])
        interpreted[i-1].append(interpreted_value)
    iiter += 1

for i in interpreted:
    i.pop(0)


sds = []
mean = []
for i in interpreted:
    sds.append(np.std(i))
    mean.append(np.mean(i))

error_plus = [a+b for a, b in zip(mean, sds)]
error_minus = [a-b for a, b in zip(mean, sds)]


print("data processing done")

x = np.linspace(0,4900, num=489)

fig, ax1 = plt.subplots()


plt.plot(x, mean, linewidth=1, color="k",label=r"$\sigma$" )


plt.plot(x, mean, linewidth=2, color="k",label=r"$\sigma$")
plt.fill_between(x, error_plus, error_minus, color="r",alpha=0.5)

ax1.set_ylabel("mx" + r" [$\frac{nM}{second}$]")
ax1.tick_params(axis="y")

ax1.set_xlabel("Time" r" ${[seconds]}$")

# dna supercoiling dynamics with dynamic cripsr control
if agents[0].control == False:
    plt.title("mRNA Noise (New Mtd. No Control)")
else:
    plt.title("mRNA Noise (New Mtd. w/ Control)")

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 20})
#plt.legend(loc='upper left', prop={'size': 20})
#plt.tight_layout()

print("done")
print("")
print("average STD: "+ str(np.mean(sds)))

plt.show()
