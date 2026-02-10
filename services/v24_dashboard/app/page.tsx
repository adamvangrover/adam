import TerminalWidget from '../TerminalWidget';
import NeuralFeedWidget from '../components/NeuralFeedWidget';
import HiveMindGrid from '../components/HiveMindGrid';
import DailyBriefing from '../components/DailyBriefing';
import MarketDashboard from '../components/MarketDashboard';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-gray-900 text-white">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Adam v24.0 Mission Control
        </p>
      </div>

      <div className="w-full mt-8">
        <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-500 mb-6">
          V30 Hive Mind Status
        </h2>
        <HiveMindGrid />
      </div>

      <div className="w-full mt-8">
        <DailyBriefing />
      </div>

      <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* System Terminal */}
        <div className="border border-gray-700 rounded p-4 h-96 bg-gray-900/50 flex flex-col">
          <h2 className="text-xl mb-4 font-bold text-cyan-500">System Terminal</h2>
          <TerminalWidget />
        </div>

        {/* Market Data Stream */}
        <div className="border border-gray-700 rounded p-4 h-96 bg-gray-900/50 flex flex-col">
          <h2 className="text-xl mb-4 font-bold text-cyan-500">Market Data</h2>
          <MarketDashboard />
        </div>
      </div>

      {/* Neural Intelligence Feed (Python Core) */}
      <div className="w-full mt-4 border border-gray-700 rounded p-4 bg-gray-900/50">
        <h2 className="text-xl mb-4 font-bold text-green-500">Neural Intelligence Feed (V30)</h2>
        <NeuralFeedWidget />
      </div>
    </main>
  )
}