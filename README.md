# VMGP
VMGP is a unified Variational auto-encoder based Multi-task Genomic Prediction model, specifically tailored for for multi-phenotype, multi-environment, and cross-population plant genomic selection (GS). The VMGP model distinguishes itself by adeptly learning the low-dimensional latent representation of genomic data. This is accomplished through self-supervised learning mechanisms that involve the compression and subsequent reconstruction of data, as facilitated by the Variational Auto-encoder (VAE) framework. This process effectively encapsulates the intrinsic information embedded within the genomic sequences.

 
Leveraging this latent genotype representation, in conjunction with a modest collection of multi-dimensional phenotypic data, the VMGP model is trained to operate an integrated multi-task phenotype prediction framework. This holistic approach not only permits an in-depth exploration of the complex interplay between genotypes and phenotypes but also synchronizes the correlations spanning multiple predictive tasks. 

 
The VMGP model has demonstrated superior performance over existing methodologies (e.g., rrBLUP and RF), across a diverse array of public datasets for maize, rice, and wheat. This comparative excellence is achieved within a unified framework with consistent parameterization, underscoring the VMGP model's versatility and adaptability.

 
The superior predictive accuracy of VMGP is a testament to its proficiency in delineating the intricate relationship between genotype and phenotype. The model's user-friendly design and universal applicability are particularly noteworthy, as it delivers robust predictive outcomes without the need for the structural redesign and parameter fine-tuning typically required in conventional deep learning tasks.


The repository for the VMGP model encompasses the source code, training methodologies, and ancillary tools, all of which are meticulously organized within the "vaegs" directory. The test dataset, specifically the wheat599, is conveniently housed in the "data" folder for easy access and utilization.

 
Furthermore, we provide a demonstrative script titled "wheat599.ipynb," which serves as a blueprint for conducting cross-validation exercises utilizing the wheat599 dataset. Researchers and practitioners are encouraged to peruse and leverage this notebook as a reference, tailoring it to their specific datasets for model validation and assessment purposes.
