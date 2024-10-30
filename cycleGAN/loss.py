import torch
from torch import nn
#from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
#import cv2
import torch.nn.functional as F
torch.manual_seed(0)


adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator
            predictions and the target labels and returns a adversarial
            loss (which you aim to minimize)
    '''


    disc_pred1 = disc_X(real_X)
    disc_loss1 = adv_criterion(disc_pred1, torch.ones_like(disc_pred1))
    disc_pred2 =disc_X(fake_X.detach())
    disc_loss2 =adv_criterion(disc_pred2, torch.zeros_like(disc_pred2))
    disc_loss = (disc_loss1 + disc_loss2)/2

    return disc_loss


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator
                  predictions and the target labels and returns a adversarial
                  loss (which you aim to minimize)
    '''


    fake_Y = gen_XY(real_X)
    disc_pred = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_pred, torch.ones_like(disc_pred))

    return adversarial_loss, fake_Y

def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity
                        loss (which you aim to minimize)
    '''

    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)

    return identity_loss, identity_X


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''

    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(real_X, cycle_X)

    return cycle_loss, cycle_X


def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator
            predictions and the true labels and returns a adversarial
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
    '''

    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adversarial_loss_A, fake_B = get_gen_adversarial_loss(real_A,disc_B,gen_AB,adv_criterion)
    adversarial_loss_B , fake_A= get_gen_adversarial_loss(real_B, disc_A,gen_BA, adv_criterion)
    adversarial_loss = adversarial_loss_A + adversarial_loss_B
    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_A, id_A = get_identity_loss(real_A,gen_BA, identity_criterion)
    identity_loss_B , id_B= get_identity_loss(real_B, gen_AB, identity_criterion)
    identity_loss = identity_loss_A + identity_loss_B
    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)

    cycle_consisteny_loss_A, cy_A = get_cycle_consistency_loss(real_A,fake_B, gen_BA,cycle_criterion)
    cycle_consisteny_loss_B, cy_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_consisteny_loss = cycle_consisteny_loss_A + cycle_consisteny_loss_B
    gen_loss = lambda_identity * identity_loss +  lambda_cycle * cycle_consisteny_loss + adversarial_loss
    # Total loss

    return gen_loss, fake_A, fake_B