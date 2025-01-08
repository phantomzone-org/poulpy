use num_bigint::BigInt;
use num_bigint::Sign;
use num_integer::Integer;
use num_traits::{Zero, One, Signed};

pub trait Div{
    fn div_floor(&self, other: &Self) -> Self;
    fn div_round(&self, other: &Self) -> Self;
}

impl Div for BigInt{

    fn div_floor(&self, other:&Self) -> Self{
        let quo: BigInt = self / other;
        if self.sign() == Sign::Minus {
            return quo - BigInt::one()
        }
        return quo
    }

    fn div_round(&self, other:&Self) -> Self{
        let (quo, mut rem) = self.div_rem(other);
        rem <<= 1;
        if rem != BigInt::zero() && &rem.abs() > other{
            if self.sign() == other.sign(){
                return quo + BigInt::one()
            }else{
                return quo - BigInt::one()
            }
        }
        return quo
    }
}

