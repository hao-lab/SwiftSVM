//
//  SVM.swift
//  SwiftSVM
//
//  Created by Wu on 14-6-13.
//  Copyright (c) 2014年 Wu. All rights reserved.
//

import Foundation

class Svm{
    
    var data=Double[][]()
    
    var labels=Double[]()
    
    var C=3.0
    var tol=1e-4
    var alphatol=1e-7
    var maxiter=10000
    var numpasses=100
    
    var N:Int
    var D:Int
    
    var alpha:Double[]
    var b=0.0
    var iter=0
    var passes=0
    
    var kernel:Kernel
    
    init(data:Double[][], labels:Double[], kernel:Kernel){
        
        self.data=data
        self.labels = labels
        
        N = data.count
        D = data[0].count
        alpha=Double[](count:N,repeatedValue:0)
        
        self.kernel=kernel
        
    }
    
    
    func randi(a:Int, b:Int) -> Int{
        var t = (Double)(arc4random() % 100000) / 100000.0 * Double(b-a) + Double(a)
        return Int(t)
    }
    
    func randf(a:Double, b:Double) -> Double{
        return (Double)(arc4random() % 100000 + 1) / 100000.0 * Double(b-a) + Double(a)
    }
    
    func marginOne(inst: Double[]) -> Double {
        var f=b
        
        for i in 0..N {
            f+=alpha[i] * labels[i] * kernel.cal(v1: inst, v2: data[i])
            
        }
        return f
    }
    
    
    //SMO
    func train() -> SvmModel {
        while passes < numpasses && iter < maxiter {
            var alphaChanged=0
            for i in 0..N {
                var Ei=marginOne(data[i]) - labels[i]
                if (labels[i]*Ei < -tol && alpha[i] < C) || (labels[i]*Ei > tol && alpha[i] > 0){
                    var j=i
                    while j==i {
                        j=randi(0,b: N)
                    }
                    var Ej=marginOne(data[j]) - labels[j]
                    
                    
                    var ai=alpha[i]
                    var aj=alpha[j]
                    var L = 0.0
                    var H = C
                    if labels[i] == labels[j]{
                        L=max(ai+aj-C, 0)
                        H=min(C,ai+aj)
                    } else {
                        L=max(aj-ai, 0)
                        H=min(C,C+aj-ai)
                    }
                    if abs(L-H) < 1e-4{
                        continue
                    }
                    var eta = 2*kernel.cal(v1: data[i], v2: data[j]) - kernel.cal(v1: data[i], v2: data[i]) - kernel.cal(v1: data[j], v2: data[j])
                    if eta >= 0{
                        continue
                    }
                    var newaj = aj - labels[j]*(Ei-Ej) / eta
                    if(newaj>H) {
                        newaj = H
                    }
                    if(newaj<L){
                        newaj = L
                    }
                    if(abs(aj - newaj) < 1e-4){
                        continue
                    }
                    alpha[j] = newaj
                    var newai = ai + labels[i]*labels[j]*(aj - newaj)
                    alpha[i] = newai
                    
                    // update the bias term
                    var b1 = b - Ei - labels[i]*(newai-ai)*kernel.cal(v1: data[i],v2: data[i]) - labels[j]*(newaj-aj)*kernel.cal(v1: data[i],v2: data[j])
                    var b2 = b - Ej - labels[i]*(newai-ai)*kernel.cal(v1: data[i],v2: data[j]) - labels[j]*(newaj-aj)*kernel.cal(v1: data[j],v2: data[j])
                    b = 0.5*(b1+b2)
                    if(newai > 0 && newai < C){
                        b = b1
                    }
                    if(newaj > 0 && newaj < C){
                        b = b2
                    }
                    
                    alphaChanged++
                }
            }
            
            iter++
            
            if alphaChanged == 0{
                passes++
            }
            else{
                passes=0
            }
        }
        var newdata=Double[][]()
        var newlabels=Double[]()
        var newalpha=Double[]()
        for i in 0..N{
            if alpha[i] > alphatol{
                newdata+=data[i]
                newlabels+=labels[i]
                newalpha+=alpha[i]
            }
        }
        
        return SvmModel(alpha:newalpha, data:newdata, labels:newlabels, b:b, kernel:kernel)
    }
}

struct Kernel{
    let kerneltype:KernelType
    let sigma:Double
    let coef0:Double
    
    init(kerneltype:KernelType, sigma:Double = 0.5, coef0:Double = 0)
    {
        self.kerneltype=kerneltype
        self.sigma=sigma
        self.coef0=coef0
    }
    
    func cal(#v1:Double[], v2:Double[]) -> Double{
        switch kerneltype{
        case .Rbf:
            return makeRBFKernel(v1:v1, v2:v2)
        case .Linear:
            return makeLinearKernel(v1:v1, v2:v2)
        default:
            return 0.0
        }
    }
    
    func makeRBFKernel(#v1:Double[], v2:Double[])->Double{
        var s=0.0
        for q in 0..v1.count{
            s = s + (v1[q] - v2[q])*(v1[q] - v2[q])
        }
        return exp(-s/(2.0*sigma*sigma))
    }
    
    func makeLinearKernel(#v1:Double[], v2:Double[]) -> Double{
        var s=0.0
        for q in 0..v1.count{
            s += v1[q] * v2[q]
        }
        return s
    }
}

struct SvmModel{
    let alpha : Double[]
    let N : Int
    let D : Int
    let data : Double[][]
    let labels : Double[]
    let kernel : Kernel
    let w = Double[]?()
    let b:Double
    
    init(alpha:Double[], data:Double[][], labels:Double[], b:Double, kernel:Kernel){
        self.alpha=alpha
        self.data=data
        self.labels=labels
        self.kernel=kernel
        N=data.count
        D=data[0].count
        self.b=b
        
        if kernel.kerneltype == KernelType.Linear{
            var xw=Double[]()
            for j in 0..D{
                var s=0.0
                for i in 0..N{
                    s += alpha[i] * labels[i] * data[i][j]
                }
                xw += s
            }
            w=xw
        } else {
            w=nil
        }
    }
    
    func marginOne(inst: Double[]) -> Double {
        var f=b
        if w != nil {
            for i in 0..D{
                f += inst[i] * w![i]
            }
        } else{
            for i in 0..N {
                f+=alpha[i] * labels[i] * kernel.cal(v1: inst, v2: data[i])
            }
        }
        return f
    }
    
    func margins(data: Double[][]) -> Double[]{
        var N=data.count
        var margins=Double[](count:N, repeatedValue:0)
        for i in 0..N{
            margins[i] = marginOne(data[i])
        }
        return margins
    }
    
    func predictOne(inst: Double[]) -> Int{
        return marginOne(inst) > 0 ? 1 : -1
    }
    func predict(data:Double[][]) -> Int[]{
        var margs = margins(data)
        var r=Int[]()
        for i in 0..margs.count{
            r += margs[i] > 0 ? 1 : -1
        }
        return r
    }
}

enum KernelType{
    case Rbf
    case Linear
    case Poly
    case Sigmoid
}
