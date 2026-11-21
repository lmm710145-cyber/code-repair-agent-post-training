"""CPU math acceptance; no GPU allocation or large model load."""
import pytest

def test_clipped_grpo_gradient_sign_and_same_policy_ratio():
    torch=pytest.importorskip('torch')
    from repairlab.grpo.objective import clipped_loss
    for advantage in (1.0,-1.0):
        new=torch.tensor([-1.0,-2.0],requires_grad=True);old=new.detach().clone()
        loss,ratio=clipped_loss(new,old,advantage,1,2,.2)
        loss.backward()
        assert torch.equal(ratio,torch.ones_like(ratio))
        assert all(g*advantage<0 for g in new.grad.tolist())


def test_clip_stops_overrewarding_large_positive_ratio():
    torch=pytest.importorskip('torch')
    from repairlab.grpo.objective import clipped_loss
    new=torch.tensor([0.],requires_grad=True);old=torch.tensor([-2.])
    loss,ratio=clipped_loss(new,old,1.,1,1,.2);loss.backward()
    assert new.grad.item()==0 and loss.item()==pytest.approx(-1.2)
