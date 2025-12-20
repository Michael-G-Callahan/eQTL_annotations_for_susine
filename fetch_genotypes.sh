#!/bin/bash
# Run this on a node with internet access and 'tabix' installed
tabix -h http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr17.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz 17:79530699-80073804 > /storage/work/mgc5166/Annotations/eQTL_annotations_for_susine/output/1kg_chr17_79530699_80073804.vcf
