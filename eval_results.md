from RecursiveMAS checkpoint, i got 

|  | math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| recursiveMAS ckpt | 78.2% | 29.67% | 23.33% | 13.33% | 26.26% | 35.19% | 3.98% |
| paper | 77.8% | 31.7% | 34% | 20% | 32.6% | 37.4% ? | 37.4% ? |


train_steps=3000, n_recursive_round = 3, latent_step=48

|  | math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outer_link | 73.8% | 27.33% | 33.33% | 16.67% | 28.79% | 32.8% | 2.18% |
| shared link (lr=5e-4) | 77% | 27.67% | 26.67% | 20% | 29.8% | 33.07% | 1.99% |
|shared link(lr=1e-3, full training samples)	|78%	|34%	|23.33%	|16.67%	|25.76%|	34.13%	|1.99%|
|shared link(lr=1e-4, full training samples)	|79.4%	|33%	|33.33%	|20%	|24.75%	|33.33%	|2.09%|
| shared-state | 77.2% | 28% | 26.67% | 23.33% | 28.28% | 32.28% | 2.27% |
| shared-tied  | 76.6% | 32.67% | 	30%	 | 13.33% | 	27.27%	| 32.01%	| 2.65%| 
|shared-state(lr=5e-4, full training samples)|	76.6%|	30.33%|	23.33%|	20%|	24.75%|	33.07%|	2.56%|
|shared-state(lr=1e-3,full training samples)	|76.4%	|29.67%	|26.67%	|23.33%	|27.78%	|31.75%	|2.27%|
|shared-state(lr=1e-4, full training samples)	|77.8%	|27.33%|	30%|	16.67%	|26.26%	|31.48%	|2.09%|

train_steps=2400, n_recursive_round = 3, latent_step=48

|  | math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outer_link | 72.6% | 31% | 23.33% | 16.67% | 22.22% | 32.54% | 2.37% |
| shared link (lr=5e-4) | 78.6% | 26.67% | 33.33% | 23.33% | 29.29% | 33.86% | 2.09% |
| shared link(lr=1e-3, full training samples)	| 78.2%	| 30.67%| 	23.33%	| 20%	| 30.81%	| 33.07%	| 1.71%| 
| shared link(lr=1e-4, full training samples)	| 75.8%	| 24.33%	| 23.33%| 	20%| 	28.28%	| 31.22%	| 1.71%| 
| shared_state | 78.4% | 33.67% | 30% | 16.67% | 24.24% | 32.54% | 2.75% |
| shared-tied	| 76.4%	| 28%	| 26.67%	| 20%	| 28.28%	| 32.01%	| 1.61%| 
|shared-state(lr=5e-4,full training samples)|	75.2%|	27%	|33.33%|	23.33%	|29.8%|	30.16%|	2.09%|
|shared-state(lr=1e-3, full training samples)	|77.2%	|30.33%|	30%	|20%	|28.28%	|31.48%	|1.42%|
|shared-state(lr=1e-4, full training samples)	|77.6%	|28%	|26.67%	|13.33%	|25.25%	|33.33%	|1.8%|
train_steps=1800, n_recursive_round = 3, latent_step=48

|  | math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outer_link | 75% | 30.67% | 30% | 13.33% | 20.71% | 34.39% | 2.37% |
| shared link (lr=5e-4) | 76.6% | 27.33% | 26.67% | 16.67% | 30.81% | 34.13% | 1.61% |
| shared link(lr=1e-3, full training samples)	| 78.2%	| 30%| 	20%	| 26.67%| 	29.29%	| 31.22%| 	1.9%| 
| shared link(lr=1e-4, full training samples)	| 77.8%	| 28.67%	| 23.33%	| 23.33%	| 29.29%	| 34.13%	| 1.33%| 
| shared_state | 77.6% | 31.67% | 26.67% | 16.67% | 33.33% | 33.86% | 2.27% |
| shared-tied	| 77.2%	| 30.00%	| 30%	| 13.33%	| 29.29%	| 31.75%	| 1.71%| 
| shared-state(lr=5e-4, full training samples)| 	77%| 	29.33%| 	30%	| 16.67%| 	33.33%| 	30.42%| 	1.8%| 
| shared-state(lr=1e-3, full training samples)	| 76.4%	| 29.33%	| 26.67%| 	16.67%| 	28.28%	| 32.54%	| 1.8%| 
| shared-state(lr=1e-4, full training samples)	| 77.6%	| 28%	| 30%| 	20%	| 30.81%	| 34.13%	| 1.99%| 

train_steps=1200, n_recursive_round = 3, latent_step=48

|  | math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outer_link | 66% | 31% | 26.67% | 13.33% | 28.79% | 34.66% | 2.09% |
| shared link (lr=5e-4) | 77.2% | 26.33% | 30% | 16.67% | 33.84% | 30.95% | 1.61% |
| shared link(lr=1e-3, full training samples)| 	77.8%	| 28%	| 30%	| 20%	| 29.8%	| 34.39%| 	1.8%| 
| shared link(lr=1e-4, full training samples)| 	77.2%| 	30.33%	| 30%	| 16.67%	| 23.74%| 	32.8%	| 1.71%| 
| shared-state | 78.6% | 30.67% | 23.33% | 20% | 29.8% | 32.54% | 2.18% |
| shared-tied	| 75.4%	| 28.33%	| 23.33%	| 13.33%	| 28.79%	| 34.66%	| 1.52%| 
| shared-state(lr=5e-4,full training samples)	| 76.8%	| 30.33%| 	30%	| 16.67%| 	30.81%| 	33.33%| 2.46%| 
|shared-state(lr=1e-3, full training samples)	|80%	|26.33%	|30%	|16.67%	|23.74%	|33.6%	|1.71%|
|shared-state(lr=1e-4, full training samples)|	78.4%|	30.67%	|20%	|13.33%	|25.25%	|34.13%	|2.27%|

train_steps=600, n_recursive_round = 3, latent_step=48

|  | math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outer_link | 65% | 24.67% | 16.67% | 13.33% | 23.23% | 32.01% | 1.23% |
| shared link(lr=5e-4) | 78.4% | 26.67% | 20% | 13.33% | 26.77% | 33.6% | 1.52% |
| shared link(lr=1e-3, full training samples)	| 78.4%	| 32%	| 30%	| 23.33%	| 30.3%	| 30.69%	| 1.9%| 
| shared link(lr=1e-4, full training samples)	| 79.2%	| 30%	| 23.33%	| 16.67%	| 33.84%	| 34.39%| 	2.09%| 
| shared-state | 79% | 29.67% | 30% | 23.33% | 29.8% | 31.22% | 2.18% |
| shared-tied	| 76.8%	| 27.67%	| 23.33%| 	13.33%	| 21.72%	| 34.13%	| 1.61%| 
| shared-state(lr=5e-4, full training samples)| 	77.4%	| 32.33%| 	26.67%| 	23.33%| 	29.8%| 	31.75%| 2.27%| 
| shared-state(lr=1e-3, full training samples)	| 77.6%	| 28.67%	| 36.67%	| 16.67%	| 29.8%	| 32.8%	| 1.9%| 
| shared-state(lr=1e-4, full training samples)	| 78.6%	| 31.67%	| 26.67%	| 23.33%	| 24.75%	| 34.39%	| 1.99%| 