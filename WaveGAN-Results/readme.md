# 🌟 SMSAT WaveGAN Results

This repository presents the **WaveGAN-based augmentation results** for the **SMSAT dataset**, including training curves, generator/discriminator architectures, and quality evaluations for all three auditory classes.

## WaveGAN Architecture

| Generator | Discriminator |
|----------|---------------|
| ![Generator](Wavegan_generator.png) | ![Discriminator](Wavegan_discriminator.png) |

## Training Loss Curves

| NS Loss | Music Loss | SM Loss |
|--------|------------|---------|
| ![NS](NS_loss_curve.png) | ![M](M_loss_curve.png) | ![SM](SM_loss_curve.png) |

## Classwise Generation Samples

| Music | Normal (Silence) | Spiritual Meditation |
|-------|------------------|----------------------|
| ![Music](quality_eval_music.png) | ![NS](quality_eval_Normal(Silence).png) | ![SM](quality_eval_SpiritualMeditation.png) |

## Signal Comparison Across All Classes

![Signal Comparison](signal_comparison_all_classes.png)

## Example Generated Output

![Generated Sample](gen1.png)

## Generated Dataset Preview

![Dataset](Generated-dataset.jpeg)

## Summary

WaveGAN successfully produced class-consistent synthetic audio for **Music**, **Normal Silence**, and **Spiritual Meditation**, supporting balanced training and improved model stability.

