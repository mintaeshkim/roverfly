from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque


def traj_segment_generator(pi, gamma, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob_vf, ob_pol = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    cur_ep_ret_detail = 0
    ep_rets_detail = []

    # Initialize history arrays
    obs_vf = np.array([ob_vf for _ in range(horizon)])
    obs_pol = np.array([ob_pol[0] for _ in range(horizon)])
    obs_pol_cnn = np.array([ob_pol[1] for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob_vf, ob_pol)
        vpred = vpred / (1 - gamma)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob_vf" : obs_vf, "ob_pol": obs_pol, "ob_pol_cnn": obs_pol_cnn, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_rets_detail": np.array(ep_rets_detail)}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

            ep_rets_detail = []
        
        i = t % horizon
        obs_vf[i] = ob_vf
        obs_pol[i] = ob_pol[0]
        obs_pol_cnn[i] = ob_pol[1]
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob_vf, ob_pol, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        
        cur_ep_ret_detail += np.array(list(_['reward_dict'].values()))
        
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            
            ep_rets_detail.append(cur_ep_ret_detail)
            cur_ep_ret_detail = 0

            ob_vf, ob_pol = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        continue_from=None,
        previous_iter=0):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space_vf = env.observation_space_vf
    ob_space_pol = env.observation_space_pol  
    ob_space_pol_cnn = env.observation_space_pol_cnn
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space_vf, ob_space_pol, ob_space_pol_cnn, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space_vf, ob_space_pol, ob_space_pol_cnn, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    ob_vf = U.get_placeholder_cached(name="ob_vf")
    ob_pol = U.get_placeholder_cached(name="ob_pol")
    ob_pol_cnn = U.get_placeholder_cached(name="ob_pol_cnn")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    # count clipped samples
    clip_frac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(1.0 - ratio), clip_param), tf.float32))

    # vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - (1 - gamma) * ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob_vf, ob_pol, ob_pol_cnn, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob_vf, ob_pol, ob_pol_cnn, ac, atarg, ret, lrmult], [losses, clip_frac])

    U.initialize()
    adam.sync()
    saver = tf.train.Saver(max_to_keep=100)

    # continue training from saved models
    if continue_from:
        latest_model = tf.train.latest_checkpoint("../tf_model/" + continue_from)
        saver.restore(tf.get_default_session(), latest_model)
        logger.log("Loaded model from {}".format(continue_from))

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, gamma, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    rewbuffer_lowerbody = deque(maxlen=100)
    rewbuffer_mvel = deque(maxlen=100)
    rewbuffer_ptrans_xy = deque(maxlen=100)
    rewbuffer_ptrans_z = deque(maxlen=100)
    rewbuffer_ptrans_vel = deque(maxlen=100)
    rewbuffer_prot_pos = deque(maxlen=100)
    rewbuffer_prot_vel = deque(maxlen=100)
    rewbuffer_torque = deque(maxlen=100)
    rewbuffer_foot_force = deque(maxlen=100)
    rewbuffer_acc = deque(maxlen=100)
    rewbuffer_footpos = deque(maxlen=100)
    rewbuffer_delta_acs = deque(maxlen=100)
    rewbuffer_upperbody = deque(maxlen=100)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True and max_iters != 1 :
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            # cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            cur_lrmult = max(1.0 - float(iters_so_far) / max_iters, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        # todo
        ob_vf, ob_pol, ob_pol_cnn, ac, atarg, tdlamret = seg["ob_vf"], seg["ob_pol"], seg["ob_pol_cnn"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob_vf=ob_vf, ob_pol=ob_pol, ob_pol_cnn=ob_pol_cnn, ac=ac, atarg=atarg, vtarg=tdlamret), deterministic=pi.recurrent)
        optim_batchsize = optim_batchsize or ob_vf.shape[0]

        pi.ob_vf_rms.update(ob_vf) # update running mean/std for policy
        pi.ob_pol_rms.update(ob_pol) # update running mean/std for policy
        pi.ob_pol_cnn_rms.update(ob_pol_cnn)

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(15, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob_vf"], batch["ob_pol"], batch["ob_pol_cnn"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(15, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        clip_fracs = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses, newclipfrac = compute_losses(batch["ob_vf"], batch["ob_pol"], batch["ob_pol_cnn"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
            clip_fracs.append(newclipfrac)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        meanclipfracs, _, _ = mpi_moments(clip_fracs, axis=0)

        logger.log(fmt_row(15, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss/"+name, lossval)
        logger.record_tabular("evaluations/ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"],
                   seg["ep_rets_detail"][:, 0],
                   seg["ep_rets_detail"][:, 1],
                   seg["ep_rets_detail"][:, 2],
                   seg["ep_rets_detail"][:, 3],
                   seg["ep_rets_detail"][:, 4],
                   seg["ep_rets_detail"][:, 5],
                   seg["ep_rets_detail"][:, 6],
                   seg["ep_rets_detail"][:, 7],
                   seg["ep_rets_detail"][:, 8],
                   seg["ep_rets_detail"][:, 9],
                   seg["ep_rets_detail"][:, 10],
                   seg["ep_rets_detail"][:, 11],
                   seg["ep_rets_detail"][:, 12]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews, rews_lowerbody, rews_mvel, rews_ptrans_xy, rews_ptrans_vel, rews_prot_pos, rews_prot_vel,\
            rews_ptrans_z, rews_torque, rews_foot_force, rews_acc, rews_footpos, rews_delta_acs, rews_upperbody\
            = map(flatten_lists, zip(*listoflrpairs))

        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        rewbuffer_lowerbody.extend(rews_lowerbody)
        rewbuffer_mvel.extend(rews_mvel)
        rewbuffer_ptrans_xy.extend(rews_ptrans_xy)
        rewbuffer_ptrans_z.extend(rews_ptrans_z)
        rewbuffer_ptrans_vel.extend(rews_ptrans_vel)
        rewbuffer_prot_pos.extend(rews_prot_pos)
        rewbuffer_prot_vel.extend(rews_prot_vel)
        rewbuffer_torque.extend(rews_torque)
        rewbuffer_foot_force.extend(rews_foot_force)
        rewbuffer_acc.extend(rews_acc)
        rewbuffer_footpos.extend(rews_footpos)
        rewbuffer_delta_acs.extend(rews_delta_acs)
        rewbuffer_upperbody.extend(rews_upperbody)

        logger.record_tabular("evaluations/meanclipfracs", meanclipfracs)
        logger.record_tabular("evaluations/EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("rewards/EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("rewards/EpRewLowerbodyPos", np.mean(rewbuffer_lowerbody))
        logger.record_tabular("rewards/EpRewMotorVel", np.mean(rewbuffer_mvel))
        logger.record_tabular("rewards/EpRewPelTransPos_xy", np.mean(rewbuffer_ptrans_xy))
        logger.record_tabular("rewards/EpRewPelTransPos_z", np.mean(rewbuffer_ptrans_z))
        logger.record_tabular("rewards/EpRewPelTransVel", np.mean(rewbuffer_ptrans_vel))
        logger.record_tabular("rewards/EpRewPelRotPos", np.mean(rewbuffer_prot_pos))
        logger.record_tabular("rewards/EpRewPelRotVel", np.mean(rewbuffer_prot_vel))
        logger.record_tabular("rewards/EpRewTorque", np.mean(rewbuffer_torque))
        logger.record_tabular("rewards/EpRewFootforce", np.mean(rewbuffer_foot_force))
        logger.record_tabular("rewards/EpRewAccel", np.mean(rewbuffer_acc))
        logger.record_tabular("rewards/EpRewFootPos", np.mean(rewbuffer_footpos))
        logger.record_tabular("rewards/EpRewDeltaAcs", np.mean(rews_delta_acs))
        logger.record_tabular("rewards/EpRewUpperbodyPos", np.mean(rewbuffer_upperbody))

        logger.record_tabular("misc/EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        env.update_iter(iters_so_far+previous_iter)
        logger.record_tabular("misc/EpisodesSoFar", episodes_so_far)
        logger.record_tabular("misc/TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("misc/TimeElapsedMin", (time.time() - tstart) / 60)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]