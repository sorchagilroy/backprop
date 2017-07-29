#!usr/bin/python

import numpy as np
import sys
import optparse
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-u",'--hiddens',dest='hidden_size', default=1,help='Number of hidden units')
optparser.add_option("-r",'--restarts',dest='num_restarts',default=100,help='Number of restarts')
(opts,_) = optparser.parse_args()

chars = ['a','b','$']
vocab_size = len(chars)
hidden_size = 7#opts.hidden_size
num_restarts=  opts.num_restarts


data = ['a']



char_int = {}
for i in range(len(chars)):
	char_int[chars[i]] = i

def softmax(y,ind):
	m = max(y)
	s= 0.0
	for i in y:
		s += np.exp(i-m)
	return np.exp(y[ind]-m)/s


def p(char,U,W,Wb,E,Eb,prev_h):
	x = [0.0 for i in range(vocab_size)]
	x[char_int[char]] = 1.0
	pre_relu_h = np.minimum(np.add(np.matmul(U,x),np.add(np.matmul(W,prev_h),Wb)),[sys.float_info.max for i in range(hidden_size)])
	h = np.maximum([0.0 for i in range(hidden_size)],pre_relu_h)
	y = np.add(np.matmul(E,h), Eb)
	ps = [softmax(y,ind) for ind in range(len(chars))]
	return ps,h

def dhdWb(h,dprev_hdWb,U,W,Wb):
	dhdWb = [[0 for i in range(hidden_size)]for i in range(hidden_size)]
	for i in range(hidden_size):
		for j in range(hidden_size):
			tot = 0.0
			if h[i] > 0:
				for k in range(hidden_size):
					tot += W[i][k]*dprev_hdWb[k][j]
				if j == i:
					tot += 1.0
			dhdWb[i][j] = tot
	return dhdWb

def dhdW(h,prev_h,dprev_hdW,U,W,Wb):
	dhdW = [[[0 for i in range(hidden_size)] for i in range(hidden_size)] for i in range(hidden_size)]
	for i in range(hidden_size):
		for j in range(hidden_size):
			for k in range(hidden_size):
				tot = 0.0
				if h[i] > 0:
					for l in range(hidden_size):
						tot += W[i][l]*dprev_hdW[l][j][k]
					if i == j:
						tot += prev_h[k]
				dhdW[i][j][k] = tot
	return dhdW


def dhdU(char,h,dprev_hdU,U,W,Wb):
	dhdU = [[[0 for i in range(vocab_size)] for i in range(hidden_size)] for i in range(hidden_size)]
	for i in range(hidden_size):
		for j in range(hidden_size):
			for k in range(vocab_size):
				tot = 0.0
				if h[i] > 0:
					for l in range(hidden_size):
						tot += W[i][l]*dprev_hdU[l][j][k]
					if i == j and k == char:
						tot += 1.0
				dhdU[i][j][k] = tot
	return dhdU


def prob_hidden(word,U,W,Wb,E,Eb):
	probs = {}
	hiddens = []
	prev_h = [0.0 for i in range(hidden_size)]
	prob = 1
	word = '$' + word
	for x in range(len(word)):
#p_x is the vector of the distribution at this time, so given what it has already seen, what are the probabilities of each of the characters. This will be useful for calculating derivatives
		p_x,prev_h = p(word[x],U,W,Wb,E,Eb,prev_h)
		probs[x] = p_x
		hiddens.append(prev_h)
		if x < len(word)-1:
			prob = prob*p_x[char_int[word[x+1]]]
		else:
			prob = prob*p_x[char_int['$']]
	return probs,hiddens,prob

#returns [dpdeb1,dpdeb2,...,dpdebn]
def dpeb(char,U,W,Wb,E,Eb,p):
	ind = char_int[char]
	out = [0 for i in range(len(p))]
	for i in range(len(p)):
		if i == ind:
			out[i] = (1-p[i])*p[i]
		else:
			out[i] = -p[i]*p[ind]
	return out


#returns dp(word)/dx for each weight x
def word_bptt(word,U,W,Wb,E,Eb):
	probs,hiddens,_ = prob_hidden(word,U,W,Wb,E,Eb)
	dpdU = [[0 for i in range(vocab_size)] for i in range(hidden_size)]
	dpdW = [[0 for i in range(hidden_size)] for i in range(hidden_size)]
	dpdWb = [0 for i in range(hidden_size)]
	dpdE = [[0 for i in range(hidden_size)] for i in range(vocab_size)]
	dpdEb = [0 for i in range(vocab_size)]
	l = len(word)+1
	word = '$' + word
	dprev_hdWb = [[0 for i in range(hidden_size)] for i in range(hidden_size)]
	dprev_hdU = [[[0 for i in range(vocab_size)] for i in range(hidden_size)] for i in range(hidden_size)]
	dprev_hdW = [[[0 for i in range(hidden_size)] for i in range(hidden_size)] for i in range(hidden_size)]
	for t in range(len(word)):
		if t < len(word) -1:
			sub_Eb = dpeb(word[t+1],U,W,Wb,E,Eb,probs[t])
			sub_E = np.outer(sub_Eb,hiddens[t])
		else:
			sub_Eb = dpeb('$',U,W,Wb,E,Eb,probs[t])
			sub_E = np.outer(sub_Eb,hiddens[t])
#P holds dp[t][current_char]/dh[t][i] for each i
		P = []
		for i in range(hidden_size):
			tot = 0.0
			if t < len(word) -1:
				for j in range(vocab_size):
					tot += probs[t][j]*(E[char_int[word[t+1]]][i]-E[j][i])
				P.append(probs[t][char_int[word[t+1]]]*tot)
			else:
				for j in range(vocab_size):
					tot += probs[t][j]*(E[char_int['$']][i] - E[j][i])
				P.append(probs[t][char_int['$']]*tot)
#dhWb is a matrix where the (i,j) element is dhjdwbi.
		dhWb = dhdWb(hiddens[t],dprev_hdWb,U,W,Wb)
		sub_Wb = [np.matmul(P,[dhWb[l][i] for l in range(hidden_size)]) for i in range(hidden_size)]
		dhU = dhdU(char_int[word[t]],hiddens[t],dprev_hdU,U,W,Wb)
		sub_U = [[np.matmul(P,[dhU[l][i][j] for l in range(hidden_size)]) for j in range(vocab_size)] for i in range(hidden_size)]
		if t == 0:
			prev = [0.0 for i in range(hidden_size)]
		else:
			prev = hiddens[t-1]
		dhW = dhdW(hiddens[t],prev,dprev_hdW,U,W,Wb)
		sub_W = [[np.matmul([dhW[l][i][j] for l in range(hidden_size)],P) for j in range(hidden_size)] for i in range(hidden_size)] 
		dprev_hdWb = dhWb
		dprev_hdU = dhU
		dprev_hdW = dhW
		for r in range(len(word)):
			if r != t:
				if r < len(word)-1:
					prob = probs[r][char_int[word[r+1]]]
				else:
					prob = probs[r][char_int['$']]
				sub_Eb = np.multiply(sub_Eb,prob)
				sub_E = np.multiply(sub_E,prob)
				sub_U = np.multiply(sub_U,prob)
				sub_W = np.multiply(sub_W,prob)
				sub_Wb = np.multiply(sub_Wb,prob)
				dpdU = np.add(dpdU, sub_U)
				dpdW = np.add(dpdW, sub_W)
				dpdWb = np.add(dpdWb, sub_Wb)
				dpdE = np.add(dpdE, sub_E)
	 			dpdEb = np.add(dpdEb, sub_Eb)
		if len(word) == 1:
			dpdU = sub_U
			dpdW = sub_W
			dpdWb = sub_Wb
			dpdE = sub_E
			dpdEb = sub_Eb
	return dpdU,dpdW,dpdWb,dpdE,dpdEb


def data_log_bptt(data,U,W,Wb,E,Eb):
	dOdU = [[0.0 for i in range(vocab_size)] for i in range(hidden_size)]
	dOdW = [[0.0 for i in range(hidden_size)] for i in range(hidden_size)]
	dOdWb = [0.0 for i in range(hidden_size)]
	dOdE = [[0.0 for i in range(hidden_size)] for i in range(vocab_size)]
	dOdEb = [0.0 for i in range(vocab_size)]
	for d in data:
		_,_,prob = prob_hidden(d,U,W,Wb,E,Eb)
		if round(prob,305) == 0.0:
			dOdU = [[-sys.float_info.max for i in range(vocab_size)]for i in range(hidden_size)]
			dOdW = [[-sys.float_info.max for i in range(hidden_size)]for i in range(hidden_size)]
			dOdWb = [-sys.float_info.max for i in range(hidden_size)]
			dOdE = [[-sys.float_info.max for i in range(hidden_size)]for i in range(vocab_size)]
			dOdEb = [-sys.float_info.max for i in range(vocab_size)]
			return dOdU,dOdW,dOdWb,dOdE,dOdEb
		dpdU,dpdW,dpdWb,dpdE,dpdEb = word_bptt(d,U,W,Wb,E,Eb)
		subU = np.multiply(dpdU,1.0/prob)
		subW = np.multiply(dpdW,1.0/prob)
		subWb = np.multiply(dpdWb, 1.0/prob)
		subE = np.multiply(dpdE,1.0/prob)
		subEb = np.multiply(dpdEb,1.0/prob)
		dOdU = np.add(dOdU, subU)
		dOdW = np.add(dOdW, subW)
		dOdWb = np.add(dOdWb,subWb)
		dOdE = np.add(dOdE,subE)
		dOdEb = np.add(dOdEb,subEb)
	return dOdU,dOdW,dOdWb,dOdE,dOdEb

# Only works in the single character case!
def consistency(U,W,Wb,E,Eb,con_length):
	count = 0
	new = 1
	string = ''
	prev_h = [0.0 for i in range(hidden_size)]
	probs,prev_h = p('$',U,W,Wb,E,Eb,prev_h)
	prob_cont = probs[0]
	prob_end = probs[1]
	old_prob_end = 2.0
	if vocab_size > 2:
		print 'Cannot calculate consistency, alphabet is too large.'
	else:
		prob = prob_end
		old_prob = -1.0
		patience = 100
		end_patience = 100
		while True:
			probs,prev_h = p('a',U,W,Wb,E,Eb,prev_h)
			current_prob = prob_cont*probs[1]
			if round(current_prob,305) == 0:
				patience -= 1
		    	else:
		    		patience = 100
		    	if patience == 0:
		    		return prob
		    	prob_cont = prob_cont*probs[0]
		   	prob_end = probs[1]
		    	if con_length < 0:
		    		if round(old_prob_end - prob_end,12) == 0:
		    			end_patience -= 1
		    		else:
		    			end_patience = 100
		    		if end_patience == 0:
		    			if round(prob_end,100) ==0:
						return prob
					prob += (current_prob*1.0/prob_end)
					return prob
			else:
				if count > con_length:
					return prob

			prob += current_prob
			count += 1
			old_prob_end = prob_end
	return prob

def initialise():
#U is a hidden*vocab size matrix
	U = [[np.random.uniform(-5,5) for e in range(vocab_size)] for e in range(hidden_size)]
#W is a hidden*hidden size matrix
	W = [[np.random.uniform(-5,5) for e in range(hidden_size)] for e in range(hidden_size)]
#Wb is a hidden size vector
	Wb = [np.random.uniform(-5,5)for e in range(hidden_size)]
#E is a vocab*hidden size matrix
	E = [[np.random.uniform(-5,5) for e in range(hidden_size)] for e in range(vocab_size)]
#Eb is a vocab size vector
	Eb = [np.random.uniform(-5,5) for e in range(vocab_size)]
	return U,W,Wb,E,Eb


num_restarts = 1
#con_length refers to the consistency length. if it's -1 we count all strings, if it's >0 we count all strings up to that length
con_length = -1
consistency_dec = 0
bestobj = -1000
total_steps = 0
eps = 0.000001

for j in range(num_restarts):
	print 'Restart number: ',j
	i = 0
	U,W,Wb,E,Eb = initialise()
	#delayed inconsistency example - make sure hidden_size = 1
#	U = [[0.0, -np.log(2)]]
#	W = [[1.0]]
#	Wb = [np.log(2)]
#	E = [[10.0],[0.0]]
#	Eb = [0, 2*np.log(2)]
	#a^nb^n example - make sure hidden_size = 7
	U = [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,-1.0,0.0]]
	W = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0,0.0],[0.0,1.0,-1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0,0.0],[0.0,1.0,-1.0,0.0,0.0,0.0,0.0]]
	Wb = [1.0,0.0,0.0,0.0,-1.0,-1.0,-1.0]
	E = [[0.0,0.0,np.log(eps),0.0,0.0,-np.log(eps),0.0],[-np.log(eps),0.0,0.0,np.log(eps),0.0,0.0,-np.log(eps)],[np.log(eps),0.0,0.0,-np.log(eps),0.0,0.0,np.log(eps)]]
	Eb = [0.0,np.log(eps),0.0]
	examples = ['a','aa','ab','ba','b','aabb','aabbb']
	for ex in examples:
		print 'string: ',ex
		_,hiddens,prob = prob_hidden(ex,U,W,Wb,E,Eb)
		print 'prob: ',prob
		for h in hiddens:
			print 'h: ',h
	#learning rate l
	l = 1
	#learning rate decay 0.5
	l_decay = 0.5
	#learning rate update frequency
	l_update = 10
	max_float = sys.float_info.max
	old_obj = 100
	prev_con = consistency(U,W,Wb,E,Eb,con_length)
	print 'Initial probability mass of first %s strings: %s'%(con_length,prev_con)
	while False:
		if i %l_update == 0:
			l = l*l_decay
		#get the derivatives
		dOdU,dOdW,dOdWb,dOdE,dOdEb = data_log_bptt(data,U,W,Wb,E,Eb)
		#update each of the weights
		U = np.add(U,np.multiply(l,dOdU))
		W = np.add(W,np.multiply(l,dOdW))
		Wb = np.add(Wb,np.multiply(l,dOdWb))
		E = np.add(E,np.multiply(l,dOdE))
		Eb = np.add(Eb,np.multiply(l,dOdEb))
		obj = 0.0
		if i%1 == 0:
			con = consistency(U,W,Wb,E,Eb,con_length)
			print 'Probability mass of first %s strings: %s'%(con_length,con)
			if round(prev_con - con,12) > 0:
			#counts the number of times the consistency decreased
				consistency_dec += 1
			prev_con = con
		for d in data:
			_,_,prob = prob_hidden(d,U,W,Wb,E,Eb)
			if round(prob,300) > 0:
				obj += np.log(prob)
			else:
				obj = -max_float
				break
		#stopping criteria
		if round(np.abs(obj-old_obj),300) == 0:
			break
		old_obj = obj
		i += 1
		if i %1000 == 0:
			print 'step ',i

	if obj > bestobj:
		bestobj = obj
		bestU = U
		bestW = W
		bestWb = Wb
		bestE = E
		bestEb = Eb
	print '\nDone in %s steps '%i
	total_steps += i
	print '(log)obj: ',obj
	print '\nFinal consistency: ',consistency(U,W,Wb,E,Eb,con_length)
	print '\nFinal weights'
	print 'U: ',U
	print 'W: ',W
	print 'Wb: ',Wb
	print 'E: ',E
	print 'Eb: ',Eb
#			for i in range(10):
#				_,_,prob = prob_hidden(''+'a'*i,U,W,Wb,E,Eb)
#				print 'prob of a^%s = %s'%(i,prob)
print 'Trained on: ',data
#	break
#	print '\n\n'
#	print 'After %s restarts.'%num_restarts
#	print 'Highest objective reached was: ',best_obj
#	print 'With weights'
#	print 'U: ',bestU[dl]
#	print 'W: ',bestW[dl]
#	print 'Wb: ',bestWb[dl]
#	print 'E: ',bestE[dl]
#	print 'Eb: ',bestEb[dl]
#	print '\n\n'
#	for i in range(10):
#		_,_,prob = prob_hidden(''+'a'*i,best_U,best_W,best_Wb,best_E,best_Eb)
#		print 'prob of a^%s= %s'%(i,prob)

print 'For probability mass of strings up to length ',con_length
print 'Across all restarts, number of times the consistency decreased: ',consistency_dec

print 'Given training data: ',data
print 'Average number of steps to convergence across 100 random restarts: ',total_steps
print 'Best objective: ',bestobj
for i in range(30):
	_,_,prob = prob_hidden(''+'a'*i,bestU,bestW,bestWb,bestE,bestEb)
	print 'prob of a^%s: %s'%(i,prob)

