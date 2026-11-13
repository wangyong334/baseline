"""Mathematical, identity and gradient contracts for explicit Hotelling readout."""
import io
import math
import unittest
from unittest import mock

import torch

from model.hotelling_readout import HotellingReadout


def observations():
    generator = torch.Generator(device="cpu").manual_seed(2037)
    n, d = 512, 4
    y = (torch.arange(n) % 2).double()
    common = torch.randn(n, d, dtype=torch.float64, generator=generator)
    h2 = common + .4 * torch.randn(n, d, dtype=torch.float64, generator=generator)
    h1 = .6 * common + .5 * torch.randn(n, d, dtype=torch.float64, generator=generator)
    h2[:, 0] += .6*y
    h1[:, 1] += .8*y
    z2 = .3*y + torch.randn(n, dtype=torch.float64, generator=generator)
    z1 = .5*y + .4*z2 + torch.randn(n, dtype=torch.float64, generator=generator)
    return (z1,z2,h1,h2),y


class HotellingReadoutTests(unittest.TestCase):
    def setUp(self):
        self.values,self.labels=observations()

    def fitted(self, variant="full"):
        head=HotellingReadout(feature_dim=4,variant=variant).double()
        report=head.fit_statistics(*self.values,self.labels)
        return head,report

    def standardized_joint(self, head):
        z1,z2,h1,h2=self.values
        raw=torch.cat((h2,z2[:,None],h1,z1[:,None]),1)
        return (raw[:,:head.active_dim]-head.raw_midpoint)/head.raw_scale

    def test_direct_score_matches_solve_and_has_unit_regularized_variance(self):
        head,report=self.fitted()
        x=self.standardized_joint(head)
        w=torch.linalg.solve(head.regularized_covariance,head.class_mean_difference)
        norm=torch.sqrt(head.class_mean_difference.dot(w))
        expected=x.mv(w)/norm
        _,info=head(*self.values,return_info=True)
        self.assertTrue(torch.allclose(info['observer_score'],expected,atol=1e-12,rtol=1e-10))
        self.assertAlmostEqual(float(head.normalized_template.dot(head.regularized_covariance.mv(head.normalized_template))),1.,places=12)
        self.assertEqual(sum(p.numel() for p in head.parameters()),2)
        self.assertFalse(report['channel_compression'])
        self.assertEqual(report['active_channels'],10)

    def test_hotelling_attains_maximum_regularized_rayleigh_quotient(self):
        head,_=self.fitted()
        delta,metric=head.class_mean_difference,head.regularized_covariance
        generator=torch.Generator(device='cpu').manual_seed(7)
        candidates=torch.randn(2000,head.active_dim,dtype=torch.float64,generator=generator)
        quotients=candidates.mv(delta).square()/(candidates.mm(metric)*candidates).sum(1)
        maximum=float(head.fisher_distance_squared)
        self.assertTrue(bool((quotients<=maximum+1e-10).all()))
        weight=head.hotelling_template
        exact=weight.dot(delta).square()/weight.dot(metric.mv(weight))
        self.assertAlmostEqual(float(exact),maximum,places=12)

    def test_signed_class_direction_flips_with_labels_not_with_score_magnitude(self):
        head,_=self.fitted()
        opposite=HotellingReadout(feature_dim=4).double()
        opposite.fit_statistics(*self.values,1.-self.labels)
        output=head(*self.values)
        self.assertGreater(float(output[self.labels==1].mean()),float(output[self.labels==0].mean()))
        self.assertTrue(torch.allclose(opposite(*self.values),-output,atol=1e-12,rtol=1e-10))

    def test_column_units_do_not_change_the_standardized_observer(self):
        head,_=self.fitted()
        z1,z2,h1,h2=self.values
        unit=torch.tensor([.1,3.,7.,.5],dtype=torch.float64)
        changed=(z1*8.,z2*.2,h1*unit,h2/unit)
        other=HotellingReadout(feature_dim=4).double()
        other.fit_statistics(*changed,self.labels)
        self.assertTrue(torch.allclose(head(*self.values),other(*changed),atol=1e-11,rtol=1e-10))

    def test_joint_score_and_snr_decompose_into_second_and_conditional_parts(self):
        head,report=self.fitted()
        x=self.standardized_joint(head)
        d=head.feature_dim+1
        residual=x[:,d:]-x[:,:d].mm(head.cross_regression.t())
        decomposed=(x[:,:d].mv(head.second_template)+residual.mv(head.conditional_template))/head.score_normalizer
        _,info=head(*self.values,return_info=True)
        self.assertTrue(torch.allclose(info['observer_score'],decomposed,atol=1e-11,rtol=1e-10))
        self.assertLess(report['fisher_decomposition_error'],1e-10)
        self.assertGreater(report['fisher_conditional_increment'],0.)

    def test_class_balance_does_not_turn_covariance_into_total_mixture_covariance(self):
        head,_=self.fitted()
        z1,z2,h1,h2=self.values
        raw=torch.cat((h2,z2[:,None],h1,z1[:,None]),1)
        class_cov=[]
        for label in (0,1):
            x=raw[self.labels==label]
            x=x-x.mean(0)
            class_cov.append(x.t().mm(x)/len(x))
        within=.5*(class_cov[0]+class_cov[1])
        self.assertTrue(torch.allclose(head.within_covariance,within/(head.raw_scale[:,None]*head.raw_scale[None,:]),atol=1e-12))
        centered=raw-raw.mean(0)
        total=centered.t().mm(centered)/len(raw)
        self.assertGreater(float((total-within).norm()),.1)

    def test_no_context_ignores_first_during_fitting_and_forward(self):
        head,_=self.fitted('no_context')
        changed=(self.values[0]*-19.,self.values[1],self.values[2]+100.,self.values[3])
        other=HotellingReadout(feature_dim=4,variant='no_context').double()
        other.fit_statistics(*changed,self.labels)
        self.assertTrue(torch.equal(head(*self.values),head(*changed)))
        self.assertEqual(float(head.raw_weight[5:].abs().sum()),0.)
        for name,value in head.named_buffers():
            self.assertTrue(torch.equal(value,dict(other.named_buffers())[name]),name)

    def test_diagonal_control_removes_covariance_cross_terms_at_same_capacity(self):
        head,_=self.fitted('no_revision')
        full,_=self.fitted()
        self.assertTrue(torch.equal(head.regularized_covariance,torch.diag(head.regularized_covariance.diag())))
        self.assertTrue(torch.equal(head.raw_midpoint,full.raw_midpoint))
        self.assertTrue(torch.equal(head.within_covariance,full.within_covariance))
        self.assertEqual(sum(p.numel() for p in head.parameters()),2)
        self.assertFalse(torch.allclose(head(*self.values),full(*self.values)))

    def test_duplicate_views_remain_one_information_dimension_not_independent_evidence(self):
        noise=torch.tensor([-1.,-.5,.5,1.],dtype=torch.float64).repeat(2)
        labels=torch.tensor([0.,0.,0.,0.,1.,1.,1.,1.],dtype=torch.float64)
        z=noise+3.*labels
        values=(z,z,z[:,None],z[:,None])
        full=HotellingReadout(feature_dim=1).double()
        single=HotellingReadout(feature_dim=1,variant='no_context').double()
        report=full.fit_statistics(*values,labels)
        single.fit_statistics(*values,labels)
        scores=torch.stack((full(*values),single(*values)),1)
        self.assertEqual(int(torch.linalg.matrix_rank(scores)),1)
        ratio=scores[:,0].dot(scores[:,1])/scores[:,1].square().sum()
        self.assertGreater(float(ratio),0.)
        self.assertTrue(torch.allclose(scores[:,0],ratio*scores[:,1],atol=1e-11))
        self.assertGreater(report['fisher_conditional_increment'],0.)
        self.assertIn('without new information',report['duplicate_observation_caveat'])

    def test_gradients_reach_both_features_logits_and_calibration(self):
        head,_=self.fitted()
        values=tuple(v.clone().requires_grad_(True) for v in self.values)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(head(*values),self.labels)
        loss.backward()
        for value in values+(head.scale_raw,head.bias):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()),0.)
        self.assertTrue(all(not v.requires_grad for v in head.buffers()))

    def test_positive_scale_is_monotone_and_unbounded_score_is_not_a_two_logit_residual(self):
        head,_=self.fitted()
        self.assertAlmostEqual(float(head.calibration_scale()),1.,places=7)
        with torch.no_grad():
            head.scale_raw.fill_(-1000.)
        self.assertGreater(float(head.calibration_scale()),0.)
        with torch.no_grad():
            head.scale_raw.fill_(10.)
            head.bias.fill_(7.)
        output,info=head(*self.values,return_info=True)
        self.assertTrue(torch.allclose(output,head.calibration_scale()*info['observer_score']+head.bias))
        self.assertGreater(float(info['delta'].abs().max()),2.)

    def test_refresh_preserves_calibration_and_forward_never_changes_statistics(self):
        head,_=self.fitted()
        with torch.no_grad():
            head.scale_raw.fill_(.3);head.bias.fill_(-.7)
        before={name:value.clone() for name,value in head.named_buffers()}
        for training in (True,False):
            head.train(training);head(*self.values)
            for name,value in head.named_buffers():self.assertTrue(torch.equal(value,before[name]),name)
        head.fit_statistics(*self.values,self.labels)
        self.assertEqual(int(head.fit_calls),2)
        self.assertAlmostEqual(float(head.scale_raw),.3,places=12)
        self.assertAlmostEqual(float(head.bias),-.7,places=12)

    def test_zero_class_difference_returns_finite_zero_score(self):
        generator=torch.Generator(device='cpu').manual_seed(37)
        half=torch.randn(32,4,dtype=torch.float64,generator=generator)
        h=half.repeat(2,1)
        z=half[:,0].repeat(2)
        labels=torch.cat((torch.zeros(32),torch.ones(32))).double()
        head=HotellingReadout(feature_dim=4).double()
        report=head.fit_statistics(z,z,h,h,labels)
        self.assertTrue(report['zero_separation'])
        self.assertTrue(torch.equal(head(z,z,h,h),torch.zeros_like(z)))
        self.assertTrue(bool(torch.isfinite(head.raw_weight).all()))

    def test_state_roundtrip_restores_fit_and_exact_outputs(self):
        head,_=self.fitted()
        with torch.no_grad():head.bias.fill_(.2)
        destination=io.BytesIO();torch.save(head.state_dict(),destination);destination.seek(0)
        restored=HotellingReadout(feature_dim=4).double()
        restored.load_state_dict(torch.load(destination,map_location='cpu'))
        self.assertTrue(restored._is_fitted)
        self.assertEqual(restored.statistics_summary(),head.statistics_summary())
        self.assertTrue(torch.equal(restored(*self.values),head(*self.values)))

    def test_invalid_data_and_nonpositive_definite_failure_leave_valid_fit_unchanged(self):
        head,_=self.fitted()
        before={name:value.clone() for name,value in head.state_dict().items()}
        with self.assertRaisesRegex(ValueError,'two training'):
            head.fit_statistics(*self.values,torch.zeros_like(self.labels))
        broken=list(self.values);broken[0]=broken[0].clone();broken[0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):head.fit_statistics(*broken,self.labels)
        with mock.patch('torch.linalg.cholesky',side_effect=RuntimeError('not positive definite')):
            with self.assertRaisesRegex(ValueError,'positive definite'):head.fit_statistics(*self.values,self.labels)
        for name,value in head.state_dict().items():self.assertTrue(torch.equal(value,before[name]),name)

    def test_unfitted_forward_fails_and_passthrough_needs_no_statistics(self):
        with self.assertRaisesRegex(RuntimeError,'training-only'):
            HotellingReadout(feature_dim=4).double()(*self.values)
        head=HotellingReadout(feature_dim=4,variant='second').double()
        self.assertTrue(torch.equal(head(*self.values),self.values[1]))
        self.assertFalse(head.fit_statistics(*self.values,self.labels)['fitted'])
        self.assertEqual(head.macs_per_pair(),0)
        self.assertEqual(HotellingReadout().macs_per_pair(),26)
        self.assertEqual(HotellingReadout(variant='no_context').macs_per_pair(),13)
        self.assertEqual(HotellingReadout().estimated_macs(10),260)


if __name__=='__main__':
    unittest.main()
