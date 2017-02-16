### Note for 
##*Deep Residual Learning For Image Recognition*

https://github.com/KaimingHe/deep-residual-networks

### Conclusions and Takeaways:
- Empirical evidence shows that residual networks are easier to optimize, and can gain accuracy from considerably increased depth.

> - VGG Nets
> - Top-1 Error Rate and Top-5 Error Rate
> 	+ For Top-1 error rate: check if the highest probability of an output is the same as the target label.
> 	+ For Top-5 error rate: check if the top five highest probabilities of an output has the target label.
> 	+ As long as the target(expected) label of an input belongs to top 5 highest probablilty predictions, we count it as a matching for the top-5 score.

##### Zero-Padding

