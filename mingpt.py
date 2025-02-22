N, E, D =range,len,True; import numpy as A
def F(x):return 0.5*x*(1+A.tanh(A.sqrt(2/A.pi)*(x+0.044715*x**3)))
def G(x):B=A.exp(x-A.max(x,axis=-1,keepdims=D));return B/A.sum(B,axis=-1,keepdims=D)
def C(x,g,b,eps=1e-5):return g*(x-A.mean(x,axis=-1,keepdims=D))/A.sqrt(A.var(x,axis=-1,keepdims=D)+eps)+b
def B(x,w,b):return x@w+b
def H(x,c_fc,c_proj):return B(F(B(x,**c_fc)),**c_proj)
def I(x,c_attn,c_proj,n_head):x=B(x,**c_attn);return B(A.hstack([G(B@C.T/A.sqrt(B.shape[-1])+(1-A.tri(x.shape[0],dtype=x.dtype))*-1e10)@D for(B,C,D)in zip(*list(map(lambda x:A.split(x,n_head,axis=-1),A.split(x,3,axis=-1))))]),**c_proj)
def J(x,mlp,attn,ln_1,ln_2,n_head):x=x+I(C(x,**ln_1),**attn,n_head=n_head);x=x+H(C(x,**ln_2),**mlp);return x
def K(i,wte,wpe,blocks,ln_f,n_head):
	A=wte[i]+wpe[range(len(i))]
	for B in blocks:A=J(A,**B,n_head=n_head)
	return C(A,**ln_f)@wte.T
def L(i,p,n,t):
	from tqdm import tqdm
	for _ in tqdm(N(t),''):i.append(int(A.argmax(K(i,**p,n_head=n)[-1])))
	return i[E(i)-t:]
def M(prompt,t=40,ms='124M',md='models'):from utils import load_encoder_hparams_and_params as D;A,B,F=D(ms,md);C=A.encode(prompt);assert E(C)+t<B['n_ctx'];G=L(C,F,B['n_head'],t);H=A.decode(G);return H
import fire;fire.Fire(M)