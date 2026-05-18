# utils.py
# Author: Alan Murphy <alanmurph94@hotmail.com>

import torch
import os
import pandas as pd
import gzip
import urllib.request
import numpy as np
import pysam
from tangermeme.utils import one_hot_encode


#taken with modification to allow N's from Tangermeme v0.2.3
def _validate_input(X, name, shape=None, dtype=None, min_value=None, 
	max_value=None, ohe=False, ohe_dim=1,allow_N=False, alphabet=['A', 'C', 'G', 'T']):
	"""An internal function for validating properties of the input.

	This function will take in an object and verify characteristics of it, such
	as the type, the datatype of the elements, its shape, etc. If any of these
	characteristics are not met, an error will be raised.


	Parameters
	----------
	X: torch.Tensor
		The object to be verified.

	name: str
		The name to reference the tensor by if an error is raised.

	shape: tuple or None, optional
		The shape the tensor must have. If a -1 is provided at any axis, that
		position is ignored.  If not provided, no check is performed. Default is
		None.

	dtype: torch.dtype or None, optional
		The dtype the tensor must have. If not provided, no check is performed.
		Default is None.
	
	min_value: float or None, optional
		The minimum value that can be in the tensor, inclusive. If None, no
		check is performed. Default is None.

	max_value: float or None, optional
		The maximum value that can be in the tensor, inclusive. If None, no
		check is performed. Default is None.

	ohe: bool, optional
		Whether the input must be a one-hot encoding, i.e., only consist of
		zeroes and ones. Default is False.
  
	allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
  		if pwm at a position is all 0's return N. Default is False.   
    
    alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor. This is used to determine the
		letters in the returned sequence. Default is the DNA alphabet. 
	"""

	if not isinstance(X, torch.Tensor):
		raise ValueError("{} must be a torch.Tensor object".format(name))

	if shape is not None:
		if len(shape) != len(X.shape):
			raise ValueError("{} must have shape {}".format(name, shape))

		for i in range(len(shape)):
			if shape[i] != -1 and shape[i] != X.shape[i]:
				raise ValueError("{} must have shape {}".format(name, shape))


	if dtype is not None and X.dtype != dtype:
		raise ValueError("{} must have dtype {}".format(name, dtype))

	if min_value is not None and X.min() < min_value:
		raise ValueError("{} cannot have a value below {}".format(name, 
			min_value))

	if max_value is not None and X.max() > max_value:
		raise ValueError("{} cannot have a value above {}".format(name,
			max_value))

	if ohe:
		values = torch.unique(X)
		if len(values) != 2:
			raise ValueError("{} must be one-hot encoded.".format(name))

		if not all(values == torch.tensor([0, 1], device=X.device)):
			raise ValueError("{} must be one-hot encoded.".format(name))
		# need to enable cases where shape is just (alphabet_size, motif_size) or 
		# (alphabet_size) when there is just 1 predicted position
		#unsqueeze and return
		if len(X.shape) <= 2 and X.shape[0] == len(alphabet):
			X = X.unsqueeze(0)
			#just one predicted position
			if len(X.shape) == 2:
				X = X.unsqueeze(2)
        # What if (batch, alphabet_size) - as in, only predicting in one position
		elif len(X.shape) == 2 and X.shape[1] == len(alphabet):
			#produce a warning, to say this is the assumption, as could also be (motif_size, alphabet_size)
			# What if (batch, alphabet_size) - as in, only predicting in one position
			print("Warning: Assuming the tensor shape is (batch, alphabet_size). If the shape is (motif_size, alphabet_size), please reformat and rerun.")
			# reformat the tensor to have shape (batch, alphabet_size, motif_size)
			X = X.unsqueeze(2)
		if ((not (X.sum(axis=1) == 1).all()) and (not allow_N)) or (
            (allow_N) and (not ((X.sum(axis=ohe_dim) == 1) | (X.sum(axis=ohe_dim) == 0)).all())):
			raise ValueError("{} must be one-hot encoded ".format(name) +
				"and cannot have unknown characters.")
            
	return X


#taken with modification to allow N's from Tangermeme v0.2.3
def characters(pwm, alphabet=['A', 'C', 'G', 'T'], force=False, allow_N=True):
	"""Converts a PWM/one-hot encoding to a string sequence.

	This function takes in a PWM or one-hot encoding and converts it to the
	most likely sequence. When the input is a one-hot encoding, this is the
	opposite of the `one_hot_encoding` function.


	Parameters
	----------
	pwm: torch.tensor, shape=(len(alphabet), seq_len)
		A numeric representation of the sequence. This can be one-hot encoded
		or contain numeric values. These numerics can be probabilities but can
		also be frequencies.

	alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor. This is used to determine the
		letters in the returned sequence. Default is the DNA alphabet.

	force: bool, optional
		Whether to force a sequence to be produced even when there are ties.
		At each position that there is a tight, the character earlier in the
		sequence will be used. Default is False.
  
	allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
  		if pwm at a position is all 0's return N. Default is True.


	Returns
	-------
	seq: str
		A string where the length is the second dimension of PWM.
	"""
	#if (batch, alphabet_size, motif_size) and batch = 1, remove batch axis
	if len(pwm.shape) == 3 and pwm.shape[0] == 1:
		pwm = pwm[0]
        
	if len(pwm.shape) != 2:
		raise ValueError("PWM must have two dimensions where the " +
			"first dimension is the length of the alphabet and the second " +
			"dimension is the length of the sequence.")

	if pwm.shape[0] != len(alphabet):
		raise ValueError("PWM must have the same alphabet size as the " +
			"provided alphabet.")

	pwm_ismax = pwm == pwm.max(dim=0, keepdims=True).values
	if pwm_ismax.sum(axis=0).max() > 1 and force == False and allow_N == False:
		raise ValueError("At least one position in the PWM has multiple " +
			"letters with the same probability.")

	alphabet = np.array(alphabet)
	if isinstance(pwm, torch.Tensor):
		pwm = pwm.numpy(force=True)

	if allow_N:
		n_inds = np.where(pwm.sum(axis=0)==0)[0]
		dna_chars = alphabet[pwm.argmax(axis=0)]
		dna_chars[n_inds] = 'N'
	else:
		dna_chars = alphabet[pwm.argmax(axis=0)]
    
	return ''.join(dna_chars)


def get_min_dtype(max_value):
    """
    Returns the minimum dtype that can hold the given max_value without overflow.
    
    Parameters:
    - max_value: The maximum value that the dtype should be able to hold.
    
    Returns:
    - The minimum dtype that can hold the given max_value.
    """
    
    if max_value <= torch.iinfo(torch.int8).max:
        return torch.int8
    elif max_value <= torch.iinfo(torch.int16).max:
        return torch.int16
    elif max_value <= torch.iinfo(torch.int32).max:
        return torch.int32
    elif max_value <= torch.iinfo(torch.int64).max:
        return torch.int64
    else:
        raise ValueError("The max_value is too large for any dtype. This will cause an overflow.")
    
       
def get_genome(build,lcl_path='./.cache',Force=False):
    """
    Downloads genome build as fasta file from UCSC, 
    if it is not already downloaded. 
    
    Parameters:
    - build: str, genome build, must be one of ['hg19','hg38']
    - lcl_path: str, optional, the local path to save the downloaded files. Default is './.cache'
    - Force: bool, optional, whether to force download the genome. Default is False.
    
    Returns:
    - gen_pth: str, path to genome fasta file
    """
    #force build to lower case
    build = build.lower()
    #ensure correct specification
    assert build in ['hg19','hg38'], "build must be one of ['hg19','hg38']"
    #download genome if not already downloaded
    gen_pth = lcl_path+"/"+build+".fa"
    
    #check if cache folder exists
    if not os.path.exists(lcl_path):
        os.makedirs(lcl_path)

    if (not os.path.exists(gen_pth)) or Force:
        if build == 'hg19':
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
        elif build == 'hg38':
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
        print(f"Downloading {build} genome...")
        urllib.request.urlretrieve(url, gen_pth+'.gz')
        # Unzip the gz file
        with gzip.open(gen_pth+'.gz', 'rb') as f_in:
            with open(gen_pth, 'wb') as f_out:
                f_out.write(f_in.read())
    return gen_pth


def pad_sequence(seq, model_receptive_field,allow_N=True):
    """
    Pads a sequence with N's if it is smaller than the model's receptive field.
    
    Parameters:
    - seq: torch.tensor, shape=(batch_size, alphabet_size, motif_size), 
        the one-hot encoded DNA sequence.
    - model_receptive_field: int, the receptive field of the genomic deep 
        learning model.
    - allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
  		if pwm at a position is all 0's return N. Default is False.    
    """
    #validate sequence & model_receptive_field
    seq = _validate_input(seq,"seq",ohe=True,allow_N=allow_N)
    
    assert model_receptive_field>0, "model_receptive_field must be greater than 0"
    
    #if smaller than receptive field based on chromosome size, pad with N's
    if seq.shape[2] < model_receptive_field:
        pad_size = model_receptive_field - seq.shape[2]
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        padding_left = torch.zeros((seq.shape[0], seq.shape[1], pad_left))
        padding_right = torch.zeros((seq.shape[0], seq.shape[1], pad_right))
        seq = torch.cat([padding_left, seq, padding_right], dim=2)
    
    return seq

def reverse_complement_dna(seq,ohe=True,allow_N=True):
    """
    Get the reverse complement of an input DNA sequence 
    whether it is character bases or one-hot encoded.
    
    Parameters:
    - seq: str or torch tensor of one-hot encoded DNA sequence.
        Dimensions: (batch_size, alphabet_size, motif_size).
    - ohe: bool, whether the input is one-hot encoded or not
    - allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
  		if pwm at a position is all 0's return N. Default is False.
    Returns:
    - rev_comp: str or torch tensor of one-hot encoded DNA sequence
    """
    if not ohe:
        seq = seq.upper()
        bases_hash = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N"
        }
        #reverse order and get complement
        rev_comp = "".join([bases_hash[s] for s in reversed(seq)])
    
    else:
        #If tensor is just (alphabet, motif_size) then add batch axis
        if len(seq.shape) == 2:
                seq = seq.unsqueeze(0)
        #input is a numpy array
        seq = _validate_input(seq,"seq",ohe=True,allow_N=allow_N)
        #reverse compliment of seq
        rev_comp = torch.flip(seq,dims=[1,2])
        
    return rev_comp


class seq_loader(object):
    """
    Class object to generate dna sequence for a genomic deep learning model.
    
    Parameters:
    - build: str, the genome build to use, "hg38" or "hg19".
    - model_receptive_field: int, the receptive field of the model
    - alphabet: list, optional, the alphabet to use for the one-hot encoding. Default is ['A', 'C', 'G', 'T'].
    """
    def __init__(self,build,model_receptive_field,alphabet=['A', 'C', 'G', 'T']):
        #force build to lower case
        build = build.lower()
        
        #ensure correct specification
        assert build in ['hg19','hg38'], "build must be one of ['hg19','hg38']"
        #make sure model's receptive field is specified
        assert model_receptive_field>0, "model_receptive_field must be greater than 0"
    
        #Get the genome data
        gen_path = get_genome(build)
        self.genome_dat = pysam.Fastafile(gen_path)
        self.model_receptive_field = model_receptive_field
        #check if any modulus of 2
        self.mod = model_receptive_field % 2
        self.alphabet = alphabet
    
    def get_chr_len(self,chrom):
        return(self.genome_dat.get_reference_length(chrom))
        
    def get_seq_start(self,chrom,seq_start,strand,
                ohe=True,rev_comp=False,pad_seq=True):
        """
        Get the DNA sequence for a given chromosome, start and end position.
        
        Parameters:
        - chrom: str, the chromosome to get the sequence from
        - seq_start: int, the start position of the sequence of interest.
        - ohe: bool, whether the input is one-hot encoded or not. Default = True.
        - rev_comp: bool, whether to get the reverse complement of the sequence. Note this 
        will not affect the strand of the gene as the returned sequence will be the reversed
        and compliment DNA returned for negative strand anyway.
        - pad_seq: bool, whether to pad the sequence with N's if the sequence is smaller than
        the model's receptive field (when TSS is at start or end of chrom). Default = True.
        
        Returns:
        - seq: str or torch.tensor, shape=(len(alphabet), seq_len), the DNA sequence
        """
        
        #work out the start and end positions
        start = max(0,seq_start)
        #if padding needed because start is at the beginning of the chromosome, get amount
        pad_N_strt = max(seq_start*-1,0)
        #max end is the chromosome length
        chrom_len = self.genome_dat.get_reference_length(chrom)
        end = min(seq_start+self.model_receptive_field+self.mod,chrom_len)
        pad_N_end = max((seq_start + self.model_receptive_field+self.mod - chrom_len),0)
        #get the sequence (with padding if necessary)
        if pad_seq:
            seq = ("N"*pad_N_strt)+self.genome_dat.fetch(chrom,start,end).upper()+("N"*pad_N_end)
        else: #no padding
            seq = self.genome_dat.fetch(chrom,start,end).upper()
        #convert to tensor if needed    
        if ohe:
            seq = one_hot_encode(seq, force=True).unsqueeze(0)
        #get strand specific gene sequence
        if strand == "-":
            seq = reverse_complement_dna(seq,ohe=ohe)
            
        if rev_comp:
            seq = reverse_complement_dna(seq,ohe=ohe)
        
        return seq
    
    def get_seq(self,chrom,gene_start,strand,
                ohe=True,rev_comp=False,pad_seq=True):
        """
        Get the DNA sequence for a given chromosome, start and end position.
        
        Parameters:
        - chrom: str, the chromosome to get the sequence from
        - gene_start: int, the start position of the TSS for the gene of interest. This 
        will be centered in the returned sequence.
        - ohe: bool, whether the input is one-hot encoded or not. Default = True.
        - rev_comp: bool, whether to get the reverse complement of the sequence. Note this 
        will not affect the strand of the gene as the returned sequence will be the reversed
        and compliment DNA returned for negative strand anyway.
        - pad_seq: bool, whether to pad the sequence with N's if the sequence is smaller than
        the model's receptive field (when TSS is at start or end of chrom). Default = True.
        
        Returns:
        - seq: str or torch.tensor, shape=(len(alphabet), seq_len), the DNA sequence
        """
        #sort issue with even model receptive fields and rev complementing to get - strand genes:
        if strand == "-" and self.model_receptive_field%2 == 0:
            #Issue is that if the model's receptive field is even, then the TSS will be in the middle + 1
            #This is fine if consistent but if the gene is on the reverse strand, then the TSS will be in the middle - 1
            #To sort this, we need to make the receptive field odd for neg stran genes
            gene_start = gene_start+1
        #work out the start and end positions
        start = max(0,gene_start - self.model_receptive_field//2)
        #if padding needed because start is at the beginning of the chromosome, get amount
        pad_N_strt = max((gene_start - self.model_receptive_field//2)*-1,0)
        #max end is the chromosome length
        chrom_len = self.genome_dat.get_reference_length(chrom)
        end = min(gene_start + (self.model_receptive_field//2)+self.mod,chrom_len)
        pad_N_end = max((gene_start + (self.model_receptive_field//2)+self.mod - chrom_len),0)
        #get the sequence (with padding if necessary)
        if pad_seq:
            seq = ("N"*pad_N_strt)+self.genome_dat.fetch(chrom,start,end).upper()+("N"*pad_N_end)
        else: #no padding
            seq = self.genome_dat.fetch(chrom,start,end).upper()
        #convert to tensor if needed    
        if ohe:
            seq = one_hot_encode(seq, force=True).unsqueeze(0)
        #get strand specific gene sequence
        if strand == "-":
            seq = reverse_complement_dna(seq,ohe=ohe)
            
        if rev_comp:
            seq = reverse_complement_dna(seq,ohe=ohe)
        
        return seq
    
    
def one_hot_encode_dna(seq: str, force:bool = False):
    """
    One-hot encode DNA sequence.
    
    Args:
        seq: DNA sequence string
        force: If True, allow non-ACGT characters
        
    Returns:
        One-hot encoded numpy array
    """
    dna_bases = np.array(['A','C','G','T'])
    seq = seq.upper()
    
    # Handle empty sequences
    if not seq:
        return np.zeros((0, len(dna_bases)))
    
    all_one_hot = []
    for seq_i in seq:
        if not force:
            if seq_i not in dna_bases:
                # If invalid base and not forcing, replace with N (encoded as all zeros)
                one_hot = np.zeros((1, len(dna_bases)))
            else:
                one_hot = np.zeros((1, len(dna_bases)))
                one_hot[0, np.where(seq_i == dna_bases)] = 1
        else:
            # Force mode - just use zeros for non-ACGT bases
            one_hot = np.zeros((1, len(dna_bases)))
            matches = np.where(seq_i == dna_bases)[0]
            if len(matches) > 0:
                one_hot[0, matches] = 1
        
        all_one_hot.append(one_hot)
    
    # Check if we have valid data to concatenate
    if not all_one_hot:
        return np.zeros((0, len(dna_bases)))
    
    try:
        all_one_hot = np.concatenate(all_one_hot)
    except Exception as e:
        # If concatenation fails, create a safe fallback
        print(f"Error in one_hot_encode_dna concatenation: {e}. Creating fallback encoding.")
        return np.zeros((len(seq), len(dna_bases)))
    
    # Only return 2d array if need to
    if all_one_hot.shape[0] == 1:
        return all_one_hot[0]
    
    return all_one_hot

def get_genome(build,lcl_path='./.cache',Force=False):
    """
    Downloads genome build as fasta file from UCSC, 
    if it is not already downloaded. 
    
    Parameters:
    - build: str, genome build, must be one of ['hg19','hg38']
    - lcl_path: str, optional, the local path to save the downloaded files. Default is './.cache'
    - Force: bool, optional, whether to force download the genome. Default is False.
    
    Returns:
    - gen_pth: str, path to genome fasta file
    """
    #force build to lower case
    build = build.lower()
    #ensure correct specification
    assert build in ['hg19','hg38','mm10'], "build must be one of ['hg19','hg38','mm10']"
    #download genome if not already downloaded
    gen_pth = lcl_path+"/"+build+".fa"
    
    #check if cache folder exists
    if not os.path.exists(lcl_path):
        os.makedirs(lcl_path)

    if (not os.path.exists(gen_pth)) or Force:
        if build == 'hg19':
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
        elif build == 'hg38':
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
        elif build == 'mm10':
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz"    
        else:
            raise ValueError(f"Invalid genome build - {build}")
        print(f"Downloading {build} genome...")
        urllib.request.urlretrieve(url, gen_pth+'.gz')
        # Unzip the gz file
        with gzip.open(gen_pth+'.gz', 'rb') as f_in:
            with open(gen_pth, 'wb') as f_out:
                f_out.write(f_in.read())
    return gen_pth

def get_metadata(organism):

    if organism == 'human':
        return {
            "num_targets": 5313,
            "train_seqs": 34021,
            "valid_seqs": 2213,
            "test_seqs": 1937,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        }
    elif organism == "mouse":
        return {
            "num_targets": 1643,
            "train_seqs": 29295,
            "valid_seqs": 2209,
            "test_seqs": 2017,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        }   

def calc_diff_basenji_enf_seqs(window,enf_wind=196_608):
    extra_bp = (enf_wind-window)//2
    return extra_bp

def extend_basenji_regions(enf_loci, organism='human',enf_wind = 196_608):
    #publicly avail regions are sized for basenji2, need to extend for enf
    extra_bp = calc_diff_basenji_enf_seqs(get_metadata(organism)["seq_length"],enf_wind=enf_wind)
    enf_loci['start_enf'] = enf_loci['start']-extra_bp
    enf_loci['end_enf'] = enf_loci['end']+extra_bp
    
    #hg38/mm10 sizes - add to positions so we can check if any are outside of the genome and pad with N's if needed
    #enf trained on X chrom too!!
    
    #https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
    hg38_sizes = pd.DataFrame({"chr":['chr'+str(i) for i in range(1,23)]+['chrX'],"chr_size":
    [248_956_422,
    242_193_529,
    198_295_559,
    190_214_555,
    181_538_259,
    170_805_979,
    159_345_973,
    145_138_636,
    138_394_717,
    133_797_422,
    135_086_622,
    133_275_309,
    114_364_328,
    107_043_718,
    101_991_189,
    90_338_345,
    83_257_441,
    80_373_285,
    58_617_616,
    64_444_167,
    46_709_983,
    50_818_468,
    156_040_895
    ]})
    
    #https://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/mm10.chrom.sizes
    mm10_sizes = pd.DataFrame({"chr":['chr'+str(i) for i in range(1,20)]+['chrX'],"chr_size":
    [195_471_971,
    182_113_224,
    160_039_680,
    156_508_116,
    151_834_684,
    149_736_546,
    145_441_459,
    129_401_213,
    124_595_110,
    130_694_993,
    122_082_543,
    120_129_022,
    120_421_639,
    124_902_244,
    104_043_685,
    98_207_768,
    94_987_271,
    90_702_639,
    61_431_566,
    171_031_299]})
    
    if organism == 'human':
        enf_loci = pd.merge(enf_loci,hg38_sizes,how="left",on="chr")
    elif organism == 'mouse':
        enf_loci = pd.merge(enf_loci,mm10_sizes,how="left",on="chr")
    else:
        raise ValueError(f"Invalid organism - {organism}")
    
    #correct for any positions that are outside of the genome        
    enf_loci["end_corrected"] = enf_loci[["end_enf","chr_size"]].apply(min, axis=1)
    
    return enf_loci

def get_enf_regions(split = None, organism = "human",enf_wind = 196_608):
    enf_loci = pd.read_csv(f"./metadata/enformer_{organism}_train_regions.bed",sep="\t",header=None)
    enf_loci.columns = ["chr", "start","end","type"]
    #number the regions to compare to tfrs
    #filter to train/valid/test if necessary
    if split is not None:
        split = split.lower()
        enf_loci = enf_loci[(enf_loci['type']==split)].reset_index(drop=True)
        enf_loci[f'{split}_ind']=enf_loci.index
        #ensure number of positions matches metadata
        assert enf_loci.shape[0]==get_metadata(organism)[f'{split}_seqs']
    else:
        org_met = get_metadata(organism)
        #create count column containing index for each row of each type
        enf_loci['ind'] = enf_loci.groupby('type').cumcount()
        assert enf_loci.shape[0]==org_met['train_seqs']+org_met['valid_seqs']+org_met['test_seqs']
    
    return extend_basenji_regions(enf_loci, organism=organism, enf_wind = enf_wind)    

# Compute Pearson correlation per channel following same approach as enformer pytorch without state accumulation
def compute_pearson_correlation(preds: torch.Tensor, target: torch.Tensor, reduce_dims=(0, 1)):
    """
    Directly compute Pearson correlation per channel without state accumulation.
    
    Args:
        preds: Prediction tensor
        target: Target tensor
        reduce_dims: Dimensions to reduce over (default: (0, 1))
        
    Returns:
        Tensor containing Pearson correlation coefficients per channel
    """
    assert preds.shape == target.shape
    
    # Calculate sums
    n = torch.prod(torch.tensor([preds.shape[dim] for dim in reduce_dims]))
    product = torch.sum(preds * target, dim=reduce_dims)
    true_sum = torch.sum(target, dim=reduce_dims)
    true_squared_sum = torch.sum(torch.square(target), dim=reduce_dims)
    pred_sum = torch.sum(preds, dim=reduce_dims)
    pred_squared_sum = torch.sum(torch.square(preds), dim=reduce_dims)
    
    # Calculate means
    true_mean = true_sum / n
    pred_mean = pred_sum / n
    
    # Calculate covariance
    covariance = (product 
                 - true_mean * pred_sum
                 - pred_mean * true_sum
                 + n * true_mean * pred_mean)
    
    # Calculate variances
    true_var = true_squared_sum - n * torch.square(true_mean)
    pred_var = pred_squared_sum - n * torch.square(pred_mean)
    
    # Calculate correlation
    tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
    correlation = covariance / tp_var
    
    return correlation    


from tangermeme.ersatz import _dinucleotide_shuffle
import numpy

#Need to reimplment this from Tangermem as the current release version (0.5.1) does not 
#allow N's in the sequence when shuffling.
def dinucleotide_shuffle(X, start=0, end=-1, n=20, random_state=None, 
    allow_N=False, verbose=False):
	"""Given a one-hot encoded sequence, dinucleotide shuffle it.

	This function takes in a one-hot encoded sequence (not a string) and
	returns a set of one-hot encoded sequences that are dinucleotide
	shuffled. The approach constructs a transition matrix between
	nucleotides, keeps the first and last nucleotide constant, and then
	randomly at uniform selects transitions until all nucleotides have
	been observed. This is a Eulerian path. Because each nucleotide has
	the same number of transitions into it as out of it (except for the
	first and last nucleotides) the greedy algorithm does not need to
	check at each step to make sure there is still a path.

	This function has been adapted to work on PyTorch tensors instead of
	numpy arrays. Code has been adapted from
	https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to be shuffled.

	start: int, optional
		The starting position of where to randomize the sequence, inclusive.
		Default is 0, shuffling the entire sequence.

	end: int, optional
		The ending position of where to randomize the sequence. If end is
		positive then it is non-inclusive, but if end is negative then it is
		inclusive. Default is -1, shuffling the entire sequence.

	n: int, optional
		The number of times to shuffle that region. Default is 20.

	random_state: int or None, optional
		Whether to use a specific random seed when generating the random insert,
		to ensure reproducibility. If None, do not use a reproducible seed.
		Unlike other methods, cannot be a numpy.random.RandomState object. 
		Default is None.
	
    allow_N: bool, optional
        Whether to allow N's in the sequence. Default is False.


	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(-1, n, k, -1)
		The shuffled sequences.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, ohe_dim=1, allow_N=allow_N)

	if end < 0:
		end = X.shape[-1] + 1 + end

	if random_state is None:
		random_state = numpy.random.randint(0, 9999999)

	X_shufs = []
	for i in range(X.shape[0]):
		insert_ = _dinucleotide_shuffle(X[i, :, start:end], n_shuffles=n, 
			random_state=random_state+i, verbose=verbose)

		X_shuf = torch.clone(X[i:i+1]).repeat(n, 1, 1)
		X_shuf[:, :, start:end] = insert_
		X_shufs.append(X_shuf)

	return torch.stack(X_shufs)


def get_gene_locations(build, lcl_path="./.cache", Force=False, filter_genes=None):
    """
    Downloads gene locations from UCSC,
    if it is not already downloaded.
    Then load and return as dataframe.

    Parameters:
    - build: str, genome build, must be one of ['hg19','hg38']
    - lcl_path: str, optional, the local path to save the downloaded files. Default is './.cache'
    - Force: bool, optional, whether to force download the genome. Default is False.
    - filter_genes: list, optional, list of genes to filter the gene locations to. Default is None.

    Returns:
    - gen_pth: str, path to genome fasta file
    """
    # force build to lower case
    build = build.lower()
    # ensure correct specification
    assert build in ["hg19", "hg38", "mm10"], (
        "build must be one of ['hg19','hg38','mm10']"
    )
    # download genome if not already downloaded
    gen_pth = lcl_path + "/" + build + "_genes.tsv"

    # check if cache folder exists
    if not os.path.exists(lcl_path):
        os.makedirs(lcl_path)

    if (not os.path.exists(gen_pth)) or Force:
        if build == "hg38":
            # url for hg38 - gencode gtf
            url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"
        elif build == "hg19":
            # not implemented
            url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh37_mapping/gencode.v44lift37.annotation.gtf.gz"
        elif build == "mm10":
            # not implemented
            raise ValueError(f"mm10 not implemented")
        else:
            raise ValueError(f"Invalid genome build - {build}")

        gen_pth_dwn = lcl_path + "/" + build + "_genes.gtf"

        print(f"Downloading {build} gene locations in TSV format...")
        urllib.request.urlretrieve(url, gen_pth_dwn + ".gz")

        # now unzip the file
        with gzip.open(gen_pth_dwn + ".gz", "rb") as f_in:
            with open(gen_pth_dwn, "wb") as f_out:
                f_out.write(f_in.read())
        # now reformat the file to tsv format
        gene_list = []
        with open(gen_pth_dwn, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if fields[2] == "gene":  # Filter transcript-level annotations
                    chrom = fields[0]
                    start = int(fields[3]) - 1  # Convert to 0-based for BED format
                    end = int(fields[4])
                    strand = fields[6]
                    gene_id = fields[8].split(";")[0].split('"')[1]
                    trans_id = fields[8].split(";")[1].split('"')[1]
                    # tss = start if strand == "+" else end  # TSS depends on strand
                    gene_list.append((gene_id, chrom, start, end, strand, trans_id))

        # Convert to Pandas DataFrame
        gene_df = pd.DataFrame(
            gene_list, columns=["gene", "chrom", "start", "end", "strand", "transcript"]
        )
        # Save as tsv file
        gene_df.to_csv(gen_pth, sep="\t", index=False)

    # read in
    gene_df = pd.read_csv(gen_pth, sep="\t")
    # filter to 'normal' chromosomes
    gene_df = gene_df[
        gene_df.chrom.isin(
            ["chr" + str(chr_i) for chr_i in range(1, 23)] + ["chrX", "chrY"]
        )
    ]

    return gene_df