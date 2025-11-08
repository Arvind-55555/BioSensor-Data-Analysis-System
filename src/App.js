import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, AlertTriangle, CheckCircle, TrendingUp, Thermometer, Droplets, Wind } from 'lucide-react';

const BioSensorAnalysis = () => {
  const [sensorData, setSensorData] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [statistics, setStatistics] = useState({});

  // Constants
  const SENSOR_RANGES = {
    pH: { min: 6.5, max: 7.8, optimal: [6.8, 7.4] },
    DO: { min: 30, max: 60, optimal: [35, 50] },
    CO2: { min: 3, max: 10, optimal: [4, 7] },
    temperature: { min: 35, max: 39, optimal: [36, 38] },
  };

  // Simulate real-time sensor data collection
  const generateSensorReading = (time) => {
    const baseValues = {
      pH: 7.0 + Math.random() * 0.4 - 0.2,
      DO: 40 + Math.random() * 10 - 5,
      CO2: 5 + Math.random() * 2 - 1,
      temperature: 37 + Math.random() * 2 - 1,
    };

    // Simulate contamination scenario (5% chance)
    if (Math.random() < 0.05) {
      baseValues.pH += Math.random() * 0.8;
      baseValues.DO -= Math.random() * 15;
      baseValues.CO2 += Math.random() * 3;
    }

    return {
      time,
      ...baseValues,
      timestamp: new Date().toLocaleTimeString(),
    };
  };

  // Anomaly Detection using Statistical Method (Z-score)
  const detectAnomalies = (data, param) => {
    if (data.length < 10) return { isAnomaly: false, score: 0 };

    const values = data.map(d => d[param]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    const latest = values[values.length - 1];
    const zScore = Math.abs((latest - mean) / stdDev);
    
    return {
      isAnomaly: zScore > 2.5,
      score: zScore,
      mean,
      stdDev
    };
  };

  // Predictive Model for Contamination Risk
  const predictContamination = (data) => {
    if (data.length < 5) return { risk: 'Low', confidence: 0, factors: [] };

    const latest = data[data.length - 1];
    const factors = [];
    let riskScore = 0;

    if (latest.pH < 6.5 || latest.pH > 7.8) {
      riskScore += 30;
      factors.push('pH out of optimal range');
    }

    if (latest.DO < 30) {
      riskScore += 25;
      factors.push('Low dissolved oxygen');
    }

    if (latest.CO2 > 8) {
      riskScore += 20;
      factors.push('High CO2 levels');
    }

    if (latest.temperature < 35 || latest.temperature > 39) {
      riskScore += 15;
      factors.push('Temperature deviation');
    }

    if (data.length >= 10) {
      const recentTrend = data.slice(-10);
      const pHtrend = recentTrend[recentTrend.length - 1].pH - recentTrend[0].pH;
      const DOtrend = recentTrend[recentTrend.length - 1].DO - recentTrend[0].DO;
      
      if (Math.abs(pHtrend) > 0.5) {
        riskScore += 10;
        factors.push('Rapid pH change detected');
      }
      if (DOtrend < -10) {
        riskScore += 15;
        factors.push('DO declining rapidly');
      }
    }

    let risk = 'Low';
    if (riskScore > 60) risk = 'High';
    else if (riskScore > 30) risk = 'Medium';

    return { risk, confidence: Math.min(riskScore, 100), factors };
  };

  // Predict Fermentation Success
  const predictFermentationSuccess = (data) => {
    if (data.length < 10) return { success: 'Unknown', probability: 0 };

    const latest = data[data.length - 1];
    let score = 100;

    Object.keys(SENSOR_RANGES).forEach(param => {
      const [min, max] = SENSOR_RANGES[param].optimal;
      const value = latest[param];
      if (value < min || value > max) {
        score -= 20;
      }
    });

    const recent = data.slice(-10);
    const pHvariance = Math.max(...recent.map(d => d.pH)) - Math.min(...recent.map(d => d.pH));
    if (pHvariance > 0.6) score -= 15;

    let success = 'Excellent';
    if (score < 50) success = 'Poor';
    else if (score < 70) success = 'Fair';
    else if (score < 85) success = 'Good';

    return { success, probability: Math.max(score, 0) };
  };

  // Calculate cell growth rate
  const estimateCellGrowth = (data) => {
    if (data.length < 10) return { rate: 'N/A', phase: 'Initial' };

    const recent = data.slice(-10);
    const avgDO = recent.reduce((sum, d) => sum + d.DO, 0) / recent.length;
    const avgpH = recent.reduce((sum, d) => sum + d.pH, 0) / recent.length;
    
    let growthRate = 0;
    let phase = 'Lag';

    if (avgDO > 35 && avgpH >= 6.8 && avgpH <= 7.4) {
      if (data.length < 20) {
        phase = 'Lag';
        growthRate = 0.1;
      } else if (data.length < 50) {
        phase = 'Exponential';
        growthRate = 0.3;
      } else if (data.length < 80) {
        phase = 'Stationary';
        growthRate = 0.05;
      } else {
        phase = 'Death';
        growthRate = -0.02;
      }
    } else {
      phase = 'Stressed';
      growthRate = -0.01;
    }

    return { rate: growthRate.toFixed(3), phase };
  };

  // Monitor and collect data
  useEffect(() => {
    let interval;
    if (isMonitoring) {
      interval = setInterval(() => {
        setSensorData(prev => {
          const newData = [...prev, generateSensorReading(prev.length)];
          return newData.slice(-100);
        });
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isMonitoring]);

  // Run analysis when data updates
  useEffect(() => {
    if (sensorData.length > 0) {
      const pHAnomaly = detectAnomalies(sensorData, 'pH');
      const DOAnomaly = detectAnomalies(sensorData, 'DO');
      const CO2Anomaly = detectAnomalies(sensorData, 'CO2');
      const tempAnomaly = detectAnomalies(sensorData, 'temperature');

      const contamination = predictContamination(sensorData);
      const fermentation = predictFermentationSuccess(sensorData);
      const growth = estimateCellGrowth(sensorData);

      setPredictions({
        contamination,
        fermentation,
        growth,
        anomalies: { pHAnomaly, DOAnomaly, CO2Anomaly, tempAnomaly }
      });

      const newAlerts = [];
      if (contamination.risk === 'High') {
        newAlerts.push({
          type: 'danger',
          message: 'High contamination risk detected!',
          time: new Date().toLocaleTimeString()
        });
      }
      if (pHAnomaly.isAnomaly || DOAnomaly.isAnomaly || CO2Anomaly.isAnomaly || tempAnomaly.isAnomaly) {
        newAlerts.push({
          type: 'warning',
          message: 'Sensor anomaly detected',
          time: new Date().toLocaleTimeString()
        });
      }

      setAlerts(prev => [...newAlerts, ...prev].slice(0, 5));

      if (sensorData.length >= 10) {
        const latest = sensorData[sensorData.length - 1];
        setStatistics({
          avgpH: (sensorData.slice(-10).reduce((sum, d) => sum + d.pH, 0) / 10).toFixed(2),
          avgDO: (sensorData.slice(-10).reduce((sum, d) => sum + d.DO, 0) / 10).toFixed(1),
          avgCO2: (sensorData.slice(-10).reduce((sum, d) => sum + d.CO2, 0) / 10).toFixed(1),
          avgTemp: (sensorData.slice(-10).reduce((sum, d) => sum + d.temperature, 0) / 10).toFixed(1),
          currentpH: latest.pH.toFixed(2),
          currentDO: latest.DO.toFixed(1),
          currentCO2: latest.CO2.toFixed(1),
          currentTemp: latest.temperature.toFixed(1)
        });
      }
    }
  }, [sensorData]);

  const getRiskColor = (risk) => {
    switch(risk) {
      case 'High': return 'text-red-600 bg-red-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'Low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getSuccessColor = (success) => {
    switch(success) {
      case 'Excellent': return 'text-green-600';
      case 'Good': return 'text-blue-600';
      case 'Fair': return 'text-yellow-600';
      case 'Poor': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
                <Activity className="text-indigo-600" size={36} />
                BioSensor Data Analysis System
              </h1>
              <p className="text-gray-600 mt-2">Real-time monitoring and predictive analytics for bioreactor health</p>
            </div>
            <button
              onClick={() => setIsMonitoring(!isMonitoring)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                isMonitoring 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
            </button>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center gap-3 mb-2">
              <Droplets className="text-blue-600" size={24} />
              <h3 className="font-semibold text-gray-700">pH Level</h3>
            </div>
            <p className="text-2xl font-bold text-gray-800">{statistics.currentpH || 'N/A'}</p>
            <p className="text-sm text-gray-500">Avg: {statistics.avgpH || 'N/A'}</p>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center gap-3 mb-2">
              <Wind className="text-cyan-600" size={24} />
              <h3 className="font-semibold text-gray-700">DO (%)</h3>
            </div>
            <p className="text-2xl font-bold text-gray-800">{statistics.currentDO || 'N/A'}</p>
            <p className="text-sm text-gray-500">Avg: {statistics.avgDO || 'N/A'}</p>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center gap-3 mb-2">
              <Activity className="text-purple-600" size={24} />
              <h3 className="font-semibold text-gray-700">CO₂ (%)</h3>
            </div>
            <p className="text-2xl font-bold text-gray-800">{statistics.currentCO2 || 'N/A'}</p>
            <p className="text-sm text-gray-500">Avg: {statistics.avgCO2 || 'N/A'}</p>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center gap-3 mb-2">
              <Thermometer className="text-red-600" size={24} />
              <h3 className="font-semibold text-gray-700">Temp (°C)</h3>
            </div>
            <p className="text-2xl font-bold text-gray-800">{statistics.currentTemp || 'N/A'}</p>
            <p className="text-sm text-gray-500">Avg: {statistics.avgTemp || 'N/A'}</p>
          </div>
        </div>

        {/* Predictions Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          {/* Contamination Risk */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <AlertTriangle size={24} />
              Contamination Risk
            </h3>
            {predictions.contamination ? (
              <>
                <div className={`inline-block px-4 py-2 rounded-full text-lg font-bold mb-3 ${getRiskColor(predictions.contamination.risk)}`}>
                  {predictions.contamination.risk}
                </div>
                <div className="mb-3">
                  <p className="text-sm text-gray-600 mb-1">Confidence Score</p>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-indigo-600 h-3 rounded-full transition-all"
                      style={{ width: `${predictions.contamination.confidence}%` }}
                    />
                  </div>
                  <p className="text-right text-sm text-gray-600 mt-1">{predictions.contamination.confidence}%</p>
                </div>
                {predictions.contamination.factors.length > 0 && (
                  <div>
                    <p className="text-sm font-semibold text-gray-700 mb-2">Risk Factors:</p>
                    <ul className="text-sm text-gray-600 space-y-1">
                      {predictions.contamination.factors.map((factor, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="text-red-500">•</span>
                          {factor}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <p className="text-gray-500">Collecting data...</p>
            )}
          </div>

          {/* Fermentation Success */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <CheckCircle size={24} />
              Fermentation Outlook
            </h3>
            {predictions.fermentation ? (
              <>
                <div className={`text-3xl font-bold mb-3 ${getSuccessColor(predictions.fermentation.success)}`}>
                  {predictions.fermentation.success}
                </div>
                <div className="mb-3">
                  <p className="text-sm text-gray-600 mb-1">Success Probability</p>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-green-600 h-3 rounded-full transition-all"
                      style={{ width: `${predictions.fermentation.probability}%` }}
                    />
                  </div>
                  <p className="text-right text-sm text-gray-600 mt-1">{predictions.fermentation.probability}%</p>
                </div>
                <p className="text-sm text-gray-600 mt-4">
                  Current conditions are {predictions.fermentation.success.toLowerCase()} for optimal fermentation.
                </p>
              </>
            ) : (
              <p className="text-gray-500">Collecting data...</p>
            )}
          </div>

          {/* Cell Growth */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <TrendingUp size={24} />
              Cell Growth Status
            </h3>
            {predictions.growth ? (
              <>
                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-1">Growth Phase</p>
                  <p className="text-2xl font-bold text-indigo-600">{predictions.growth.phase}</p>
                </div>
                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-1">Growth Rate (μ)</p>
                  <p className="text-2xl font-bold text-gray-800">{predictions.growth.rate} h⁻¹</p>
                </div>
                <div className="bg-indigo-50 rounded-lg p-3 mt-4">
                  <p className="text-sm text-gray-700">
                    <strong>Phase Info:</strong> The {predictions.growth.phase} phase is characterized by 
                    {predictions.growth.phase === 'Exponential' && ' rapid cell division and biomass accumulation.'}
                    {predictions.growth.phase === 'Lag' && ' adaptation to growth medium.'}
                    {predictions.growth.phase === 'Stationary' && ' balanced growth and death rates.'}
                    {predictions.growth.phase === 'Death' && ' declining viable cell count.'}
                    {predictions.growth.phase === 'Stressed' && ' suboptimal growth conditions.'}
                  </p>
                </div>
              </>
            ) : (
              <p className="text-gray-500">Collecting data...</p>
            )}
          </div>
        </div>

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">Recent Alerts</h3>
            <div className="space-y-2">
              {alerts.map((alert, idx) => (
                <div 
                  key={idx}
                  className={`p-3 rounded-lg flex items-center gap-3 ${
                    alert.type === 'danger' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'
                  }`}
                >
                  <AlertTriangle size={20} />
                  <span className="flex-1">{alert.message}</span>
                  <span className="text-sm">{alert.time}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Real-time Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">pH & Temperature Trends</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={sensorData.slice(-30)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="pH" stroke="#3b82f6" strokeWidth={2} dot={false} />
                <Line yAxisId="right" type="monotone" dataKey="temperature" stroke="#ef4444" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">DO & CO₂ Levels</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={sensorData.slice(-30)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="DO" stroke="#06b6d4" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="CO2" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Data Summary */}
        {sensorData.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">System Information</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Total Readings</p>
                <p className="text-xl font-bold text-gray-800">{sensorData.length}</p>
              </div>
              <div>
                <p className="text-gray-600">Monitoring Status</p>
                <p className="text-xl font-bold text-green-600">{isMonitoring ? 'Active' : 'Paused'}</p>
              </div>
              <div>
                <p className="text-gray-600">Last Update</p>
                <p className="text-xl font-bold text-gray-800">{sensorData[sensorData.length - 1]?.timestamp || 'N/A'}</p>
              </div>
              <div>
                <p className="text-gray-600">Alert Count</p>
                <p className="text-xl font-bold text-red-600">{alerts.length}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BioSensorAnalysis;